import os
import shutil
import json
from glob import glob
from tqdm import tqdm

# ================= 參數設定區 (請確認這裡) =================
#    結構應該是: ./kits23/dataset/case_00000/imaging.nii.gz
source_dir = "./kits23/dataset" 

# 2. 你想要輸出的目標資料夾 (Medical-SAM-Adapter 要讀這個)
target_dir = "./dataset/KiTS23_for_MSA"

# ========================================================

def create_kits_for_msa():
    # 建立目標資料夾結構
    images_out = os.path.join(target_dir, "imagesTr")
    labels_out = os.path.join(target_dir, "labelsTr")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    print(f"🚀 開始轉換 KiTS23 資料...")
    print(f"   來源: {source_dir}")
    print(f"   目標: {target_dir}")

    # 搜尋所有病例資料夾
    case_folders = sorted(glob(os.path.join(source_dir, "case_*")))
    
    if len(case_folders) == 0:
        print("❌ 錯誤: 找不到任何 case 資料夾！請檢查 source_dir 路徑是否正確。")
        return

    data_list = []
    
    # 使用 tqdm 顯示進度條
    for case_path in tqdm(case_folders, desc="Processing"):
        case_name = os.path.basename(case_path) # e.g., "case_00000"
        
        # 定義原始檔案路徑
        src_img = os.path.join(case_path, "imaging.nii.gz") # 原始 CT
        src_seg = os.path.join(case_path, "segmentation.nii.gz") # 標籤 mask
        
        # 確認影像和標籤都存在才處理
        if os.path.exists(src_img) and os.path.exists(src_seg):
            # 定義新的檔案名稱 (保持 .nii.gz 壓縮格式)
            dst_filename = f"{case_name}.nii.gz"
            
            dst_img_path = os.path.join(images_out, dst_filename)
            dst_seg_path = os.path.join(labels_out, dst_filename)
            
            # 複製檔案 (Copy)
            # 如果目標檔案已經存在，就跳過 (節省時間)
            if not os.path.exists(dst_img_path):
                shutil.copy2(src_img, dst_img_path)
            
            if not os.path.exists(dst_seg_path):
                shutil.copy2(src_seg, dst_seg_path)
            
            # 加入到列表，準備寫入 JSON
            # 注意：這裡使用相對路徑，這是 Medical-SAM-Adapter 的標準寫法
            data_list.append({
                "image": f"./imagesTr/{dst_filename}",
                "label": f"./labelsTr/{dst_filename}"
            })

    print(f"✅ 檔案複製完成！共處理 {len(data_list)} 筆資料。")

    # === 生成 dataset.json ===
    # 自動切分訓練集和驗證集 (例如：最後 20% 做驗證，或者固定數量)
    # 這裡示範：最後 10 筆當驗證集，其餘訓練
    val_count = 10 
    if len(data_list) > val_count:
        train_data = data_list[:-val_count]
        val_data = data_list[-val_count:]
    else:
        # 如果資料很少，就全部當訓練
        train_data = data_list
        val_data = []

    json_output = {
        "name": "KiTS23",
        "description": "Kidney Tumor Segmentation Challenge 2023",
        "tensorImageSize": "3D",
        "modality": {"0": "CT"},
        "labels": {
            "0": "background",
            "1": "kidney", 
            "2": "tumor", 
            "3": "cyst"
        },
        "numTraining": len(train_data),
        "numValidation": len(val_data),
        "training": train_data,
        "validation": val_data
    }

    json_path = os.path.join(target_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=4)

    print(f"📄 dataset.json 已生成於: {json_path}")
    print(f"🎉 準備就緒！請使用 -data_path {target_dir} 進行訓練。")

if __name__ == "__main__":
    create_kits_for_msa()