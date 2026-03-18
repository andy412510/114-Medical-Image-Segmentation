import sys
import torch
import argparse
import os
from tqdm import tqdm

# --- 1. 參數提取與 sys.argv 修正 ---
# 為了避開 cfg.py 的參數檢查，手動提取 --ckpt 並從系統參數移除
ckpt_path = None
if '--ckpt' in sys.argv:
    idx = sys.argv.index('--ckpt')
    ckpt_path = sys.argv[idx + 1]
    del sys.argv[idx:idx+2]

parser = argparse.ArgumentParser(description='Med-SA 3D Inference Script')
parser.add_argument('--sam_ckpt', type=str, default='./checkpoint/sam/sam_vit_b_01ec64.pth')
parser.add_argument('--data_path', type=str, default='/home/user412771213/folder/data')
parser.add_argument('--gpu_device', type=int, default=0)
parser.add_argument('--roi_size', type=int, default=96)
parser.add_argument('--chunk', type=int, default=2) # 對應 Adapter 的 depth (d)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--out_size', type=int, default=256)

args, unknown = parser.parse_known_args()

# --- 2. 補齊模型構建所需的屬性 (必須與訓練時一致) ---
args.ckpt = ckpt_path
args.mod = 'sam_adpt'        
args.net = 'sam'             
args.multimask_output = 1    
args.thd = True              
args.mid_dim = 768             
args.patch_size = 16         
args.dim = 768               

if args.ckpt is None:
    print("錯誤：必須提供 --ckpt 參數")
    sys.exit(1)

# --- 3. 載入必要模組 ---
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImaged, ScaleIntensityRanged, CropForegroundd, Orientationd, Spacingd, EnsureTyped
from monai.data import CacheDataset, ThreadDataLoader, load_decathlon_datalist
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from models.sam import sam_model_registry
import function # 這裡會跑 cfg，但因為 --ckpt 已被移除，所以不會噴錯

device = torch.device(f'cuda:{args.gpu_device}')

# --- 4. 初始化模型與載入權重 ---
net = sam_model_registry['default'](args, checkpoint=args.sam_ckpt).to(device)
checkpoint = torch.load(args.ckpt, map_location=device)
net.load_state_dict(checkpoint['state_dict'])
net.eval()
print(f"模型載入完成：{args.ckpt}")

# --- 5. 核心 Wrapper：處理 3D Adapter 的 Batch/Depth 要求 ---
def sam_forward_wrapper(inputs, **kwargs):
    b, c, h, w, d = inputs.shape # 如果 chunk=1, d 就會是 1
    
    # 這裡會變成 (b*1, c, h, w)
    inputs_2d = inputs.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
    
    if inputs_2d.shape[1] == 1:
        inputs_2d = inputs_2d.repeat(1, 3, 1, 1)
        
    batched_input = [{"image": inputs_2d[i], "original_size": (args.image_size, args.image_size)} for i in range(inputs_2d.shape[0])]
    
    results = net(batched_input, multimask_output=kwargs.get("multimask_output", False))
    
    masks = results[0]['masks'] if isinstance(results, list) else results
    if masks.dtype == torch.bool:
        masks = masks.float()
        
    if masks.shape[-2:] != (h, w):
        masks = torch.nn.functional.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        
    out_c = masks.shape[1]
    # 因為 b=1, d=1，這裡 reshape 就會變成 (1, 1, out_c, 96, 96)，絕對不會出錯
    return masks.reshape(b, d, out_c, h, w).permute(0, 2, 3, 4, 1).contiguous()

# --- 6. 資料流準備 ---
val_transforms = Compose([
    LoadImaged(keys=["image", "label"], ensure_channel_first=True),
    ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    EnsureTyped(keys=["image", "label"], device=device),
])

data_dir = args.data_path
datasets_json = os.path.join(data_dir, "dataset_0.json")
val_files = load_decathlon_datalist(datasets_json, True, "validation")

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=10)
val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

# 針對 BTCV 14 類分割
dice_metric = DiceMetric(include_background=True, reduction="mean")
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)

# --- 7. 開始測試 ---
print(f"開始測試 validation set (共 {len(val_files)} 個案例)...")
dice_metric.reset()
with torch.no_grad():
    for batch_data in tqdm(val_loader):
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        # 呼叫滑窗推論，傳入包裝後的 sam_forward_wrapper
        outputs = sliding_window_inference(
            inputs, 
            (args.roi_size, args.roi_size, args.chunk), 
            4, 
            sam_forward_wrapper,
            multimask_output=(args.multimask_output > 1)
        )

        outputs = post_pred(outputs)
        labels = post_label(labels)
        dice_metric(y_pred=outputs, y=labels)

    mean_dice = dice_metric.aggregate().item()
    print(f"\n平均 Dice 分數 (14 類): {mean_dice:.4f}")

print("測試流程結束！")