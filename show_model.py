import torch
from torchinfo import summary
from models.sam.build_sam import sam_model_registry
import cfg

# python show_model.py -net sam -mod sam_adpt


def main():
    """
    This script prints the model architecture of the 'vit_b' model from SAM.
    """
    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)
    
    # Create a dummy args object and set necessary attributes

    # Get the 'vit_b' model from the registry
    model = sam_model_registry['vit_b'](args=args, checkpoint=None).to(GPUdevice)
    print("Model built successfully.")
    print("-" * 80)
    image_encoder = model.image_encoder
    input_size = (2, 3, args.image_size, args.image_size)
    print(f"Model Summary for 'vit_b' IMAGE ENCODER with input size {input_size}:")
    summary(
        image_encoder,
        input_size=input_size,
    )
    print("-" * 80)


if __name__ == '__main__':
    main()