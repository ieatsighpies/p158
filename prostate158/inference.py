import os
import torch
import monai
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    NormalizeIntensity,
    Spacing,
    Orientation,
    SaveImage,
    Compose,
    ConcatItemsd,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    NormalizeIntensityd,
    Spacingd,
    Orientationd,
    SpatialCropd,
    CenterSpatialCropd,
)
from .model import get_model
from .utils import load_config


def load_pretrained_model(config, checkpoint_path):
    """Load the pretrained model"""
    model = get_model(config).to(config.device)
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    model.eval()
    return model


def get_transforms(config):
    """Get basic preprocessing transforms"""
    # Create dictionary transforms since we have multiple inputs
    transforms = [
        LoadImaged(keys=config.data.image_cols),
        EnsureChannelFirstd(keys=config.data.image_cols),
    ]

    if config.transforms.spacing:
        transforms.append(
            Spacingd(
                keys=config.data.image_cols,
                pixdim=config.transforms.spacing,
                mode="bilinear",
            )
        )

    if config.transforms.orientation:
        transforms.append(
            Orientationd(
                keys=config.data.image_cols, axcodes=config.transforms.orientation
            )
        )

    # Add center spatial crop to ensure consistent dimensions
    transforms.append(
        CenterSpatialCropd(
            keys=config.data.image_cols,
            roi_size=(64, 64, 64),  # Use the same size as in config
        )
    )

    transforms.extend(
        [
            ScaleIntensityd(keys=config.data.image_cols, minv=0, maxv=1),
            NormalizeIntensityd(keys=config.data.image_cols),
        ]
    )

    return monai.transforms.Compose(transforms)


def inference_pipeline(
    t2_path,
    adc_path,
    dwi_path,
    output_path,
    config_path="tumor.yaml",
    checkpoint_path="models/tumor.pt",
):
    """Run inference on a single case with multiple input channels

    Args:
        t2_path: Path to T2W image (NIfTI format)
        adc_path: Path to ADC image (NIfTI format)
        dwi_path: Path to DWI image (NIfTI format)
        output_path: Path where to save the output segmentation
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
    """
    # Load config
    config = load_config(config_path)
    if torch.cuda.is_available():
        config.device = "cuda:0"
    else:
        config.device = "cpu"

    # Load model
    model = load_pretrained_model(config, checkpoint_path)

    # Setup transforms
    transforms = get_transforms(config)

    # Create input dictionary
    input_dict = {"t2": t2_path, "adc": adc_path, "dwi": dwi_path}

    # Load and preprocess images
    print(f"Processing images...")
    data = transforms(input_dict)

    # Stack the channels correctly
    # Each image should be [C, D, H, W]
    t2 = data["t2"]  # Should already be in correct format from transforms
    adc = data["adc"]
    dwi = data["dwi"]

    # Stack along channel dimension
    image = torch.cat([t2, adc, dwi], dim=0)  # [3, D, H, W]

    # Add batch dimension
    image = image.unsqueeze(0).to(config.device)  # [1, 3, D, H, W]

    print(f"Input shape: {image.shape}")

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(image)
        # Apply softmax for multi-class segmentation
        output = torch.softmax(output, dim=1)
        # Get the tumor class (class 1)
        tumor_pred = output[:, 1:2]
        # Threshold at 0.5
        tumor_pred = (tumor_pred > 0.5).float()

    # Save output
    print(f"Saving output to: {output_path}")
    tumor_pred = tumor_pred.squeeze().cpu()
    saver = SaveImage(
        output_dir=os.path.dirname(output_path),
        output_postfix="",
        output_ext=".nii.gz",
        separate_folder=False,
    )
    saver(tumor_pred)
