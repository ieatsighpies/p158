#!/usr/bin/env python
# coding: utf-8

# # Segmentation Example
# 
# > Train a U-Net for pixelwise segmentation of the prostate
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# # if you have a requirements.txt:
# !pip install --upgrade pip
# !pip install -r requirements.txt

# # Otherwise install core libs directly:
# !pip install monai["all"] ignite matplotlib pyyaml munch


# In[3]:


# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install opencv-python
# !pip install ipywidgets


# In[2]:


import monai
import ignite
import torch

from monai.data.meta_tensor import MetaTensor

# Add MetaTensor to the safe globals list
torch.serialization.add_safe_globals([MetaTensor])


# # Configure torch serialization to handle METATENSOR
# ts._use_global_metatensor = True  # Ensure METATENSOR is accepted globally
# ts._weight_only = False  # Disable weight-only serialization for smooth unpickling

from prostate158.utils import load_config
from prostate158.train import SegmentationTrainer
from prostate158.report import ReportGenerator
from prostate158.viewer import ListViewer
import prostate158.utils as utils
from prostate158.utils import load_config
import psutil
import subprocess
import os


# In[3]:


print(torch.cuda.is_available())  # For PyTorch


# In[ ]:


pip list 


# In[4]:


# 0) Helper to print system + GPU memory
def print_memory_stats(stage=""):
    # System RAM
    mem = psutil.virtual_memory()
    print(
        f"\n[MEMORY] {stage} ▶ System RAM: "
        f"total {mem.total/1e9:.1f} GB, used {mem.used/1e9:.1f} GB ({mem.percent}%)"
    )
    # GPU RAM (if available)
    if torch.cuda.is_available():
        # call nvidia-smi
        print("[MEMORY] GPU status via nvidia-smi:")
        try:
            gpu_info = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used",
                    "--format=csv",
                ]
            ).decode("utf-8")
            print(gpu_info.strip())
        except Exception as e:
            print("  (nvidia-smi failed:", e, ")")
        # PyTorch peak stats
        torch.cuda.reset_peak_memory_stats()
    print()


# In[ ]:


import os
import nibabel as nib


def check_nifti_sizes(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                try:
                    img = nib.load(file_path)
                    print(f"File: {file_path}")
                    print(f"Shape: {img.shape}")
                    print(f"Header: {img.header}")
                    print("-" * 50)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")


if __name__ == "__main__":
    train_dir = os.path.join(os.getcwd(), "prostate158", "train")
    check_nifti_sizes(train_dir)


# In[ ]:


import nibabel as nib

img = nib.load("prostate158/train/065/t2.nii.gz")
print(f"Image shape: {img.shape}")
print(f"Pixel dimensions (mm): {img.header.get_zooms()}")


# In[ ]:


import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Path to your files
base_path = r"C:\Users\anson\Google Drive\00_CAREER AND EDUCATION\00_SGH Career\Prostate Cancer Project\prostate158-main\prostate158\train\060"
t2_path = f"{base_path}/t2.nii.gz"
tumor_mask_path = f"{base_path}/t2_tumor_reader1.nii.gz"

# Load the files
t2_img = nib.load(t2_path)
tumor_mask = nib.load(tumor_mask_path)

# Get the data arrays
t2_data = t2_img.get_fdata()
tumor_data = tumor_mask.get_fdata()

# Visualize middle slices
slice_idx = t2_data.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# T2 image
axes[0].imshow(t2_data[:, :, slice_idx].T, cmap="gray", origin="lower")
axes[0].set_title("T2 MRI")
axes[0].axis("off")

# Tumor mask
axes[1].imshow(tumor_data[:, :, slice_idx].T, cmap="Reds", origin="lower")
axes[1].set_title("Tumor Mask")
axes[1].axis("off")

# Overlay
axes[2].imshow(t2_data[:, :, slice_idx].T, cmap="gray", origin="lower")
axes[2].imshow(tumor_data[:, :, slice_idx].T, cmap="Reds", alpha=0.5, origin="lower")
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# Print metadata
print("\nT2 Image Metadata:")
print(f"Shape: {t2_data.shape}")
print(f"Voxel sizes (mm): {t2_img.header.get_zooms()}")
print(f"Data type: {t2_data.dtype}")

print("\nTumor Mask Metadata:")
print(f"Shape: {tumor_data.shape}")
print(
    f"Unique values: {np.unique(tumor_data)}"
)  # Should be 0 (background) and 1 (tumor)
print(f"Voxel sizes: {tumor_mask.header.get_zooms()}")


# In[ ]:


help(monai.networks.nets)


# In[ ]:


# cfg = load_config("anatomy.yaml")
# print("Anatomy train CSV:", cfg.data.train_csv)
# print("Data directory    :", cfg.data.data_dir)


# All parameters needed for training and evaluation are set in `anatomy.yaml` file.
# 

# In[5]:


# config = load_config("tumor.yaml")  # change to 'tumor.yaml' for tumor segmentation
# monai.utils.set_determinism(seed=config.seed)


# In[5]:


from prostate158.utils import load_config
from prostate158.model import get_model
import torch  # Import torch to check for CUDA availability

# force‐reload the updated module
cfg = load_config("tumor.yaml")
monai.utils.set_determinism(seed=cfg.seed)
# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    cfg.device = "cuda"
else:
    cfg.device = "cpu"  # Fallback to CPU if no GPU is available

cfg.model.type = "rrunet3d"
model = get_model(cfg).to(cfg.device)  # Move the model to the selected device
print(model)


# In[7]:


cfg.model


# In[6]:


cfg.device


# In[ ]:


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Move your model to the device first
model = model.to(device)

# 2. Then move your input tensor to the same device
x = torch.randn(1, 3, 64, 128, 128).to(device)  # dummy 3D batch

# Now this should work
y = model(x)
print(y.shape)


# In[ ]:


y


# ---
# 
# **test**
# 

# In[ ]:


import torch
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
)
from monai.inferers import sliding_window_inference
from prostate158.utils import load_config
from prostate158.model import get_model

# 1) load config & model
cfg = load_config("tumor.yaml")
cfg.device = "cpu"
model = get_model(cfg).to(cfg.device)
model.eval()

# 2) same pre_transforms as before
pre_transforms = Compose(
    [
        LoadImaged(keys=cfg.data.image_cols),
        EnsureChannelFirstd(keys=cfg.data.image_cols),
        Orientationd(keys=cfg.data.image_cols, axcodes="RAS"),
        Spacingd(
            keys=cfg.data.image_cols,
            pixdim=cfg.transforms.spacing,
            mode=tuple("bilinear" for _ in cfg.data.image_cols),
        ),
        ScaleIntensityRanged(
            keys=cfg.data.image_cols,
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=cfg.data.image_cols),
    ]
)

# 3) Point to each modality’s file for one patient
#
sample_files = {
    "t2": f"C:/Users/anson/Google Drive/00_CAREER AND EDUCATION/00_SGH Career/Prostate Cancer Project/prostate158-main/prostate158/train/020/t2.nii.gz",
    "adc": f"C:/Users/anson/Google Drive/00_CAREER AND EDUCATION/00_SGH Career/Prostate Cancer Project/prostate158-main/prostate158/train/020/adc.nii.gz",
    "dwi": f"C:/Users/anson/Google Drive/00_CAREER AND EDUCATION/00_SGH Career/Prostate Cancer Project/prostate158-main/prostate158/train/020/dwi.nii.gz",
}

sample = pre_transforms(sample_files)

# 4) cat into (1, C, D, H, W)
img = torch.cat([sample[k] for k in cfg.data.image_cols], dim=0)
img = img.unsqueeze(0).to(cfg.device)

# 5) sliding-window inference over, say, 96³ patches
with torch.no_grad():
    logits = sliding_window_inference(
        inputs=img,
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,  # 25% overlap between windows
    )  # → (1, out_ch, D, H, W)

# 6) convert & plot as before
probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
pred = probs.argmax(axis=0)
mid = pred.shape[0] // 2
t2w = sample[cfg.data.image_cols[0]].cpu().numpy()[0, mid]

plt.figure(figsize=(6, 6))
plt.imshow(t2w, cmap="gray")
plt.contour(pred[mid], levels=[0.5], colors="r")
plt.axis("off")
plt.show()


# ---
# 

# In[8]:


# 4) Print memory before training
print_memory_stats("Before training")


# Create supervised trainer for segmentation task
# 

# In[9]:


trainer = SegmentationTrainer(
    progress_bar=True,
    early_stopping=True,
    metrics=["MeanDice", "HausdorffDistance", "SurfaceDistance"],
    save_latest_metrics=True,
    config=cfg,
)

%load_ext tensorboard
%tensorboard --logdir=$config.log_dir
# Adding a learning rate scheduler for one-cylce policy.
# 

# In[10]:


trainer.fit_one_cycle()


# In[11]:


# 6) Print peak GPU memory after fit_one_cycle
if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[MEMORY] Peak GPU memory used during fit_one_cycle: {peak:.2f} GB")


# Let's train. This can take several hours.
# 

# In[11]:


# # Create a comprehensive fix for the KeyError
# import types


# def robust_meta_dict(self, batch):
#     """Extract metadata from any available source in the batch"""
#     # Check what keys are available in the first batch item
#     print(
#         "Available batch keys:",
#         list(batch[0].keys()) if batch and len(batch) > 0 else "Empty batch",
#     )

#     # Try different possible metadata sources
#     possible_keys = [
#         "image_meta_dict",
#         "label_meta_dict",
#     ]  # Standard MONAI keys after concatenation

#     # Also try keys based on original image columns
#     image_cols = self.config.data.image_cols
#     if isinstance(image_cols, (list, tuple)):
#         for col in image_cols:
#             possible_keys.append(f"{col}_meta_dict")
#     else:
#         possible_keys.append(f"{image_cols}_meta_dict")

#     # For each item in batch, find first available metadata or create default
#     result = []
#     for item in batch:
#         # Debug info
#         if len(result) == 0:  # Only print for first item to avoid flooding
#             print(f"Item keys: {list(item.keys())}")

#         # Find the first available metadata key
#         meta = None
#         for key in possible_keys:
#             if key in item:
#                 meta = item[key]
#                 if len(result) == 0:  # Only print for first item
#                     print(f"Found metadata in key: {key}")
#                 break

#         # If no metadata found, create minimal default
#         if meta is None:
#             print("No metadata found, creating default")
#             meta = {
#                 "filename_or_obj": "unknown",
#                 "spatial_shape": [96, 96, 96],
#                 "original_shape": [96, 96, 96],
#             }

#         result.append(meta)

#     return result


# # Create a function to inspect batch structure
# def inspect_batch(batch):
#     """Print the structure of a batch"""
#     print("\n--- BATCH INSPECTION ---")
#     print(f"Batch type: {type(batch)}")
#     print(f"Batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")

#     if hasattr(batch, "__getitem__") and len(batch) > 0:
#         item = batch[0]
#         print(f"First item type: {type(item)}")
#         if hasattr(item, "keys"):
#             print(f"First item keys: {list(item.keys())}")

#             # Check metadata keys specifically
#             meta_keys = [k for k in item.keys() if "meta" in k]
#             print(f"Metadata keys: {meta_keys}")

#             # Print first metadata structure if available
#             if meta_keys and meta_keys[0] in item:
#                 print(f"Structure of {meta_keys[0]}:")
#                 meta = item[meta_keys[0]]
#                 print(f"  Type: {type(meta)}")
#                 if hasattr(meta, "keys"):
#                     print(f"  Keys: {list(meta.keys())}")

#     print("--- END INSPECTION ---\n")
#     return batch


# # Add inspection to trainer
# import monai.engines

# # Save original _get_meta_dict
# original_get_meta_dict = SegmentationTrainer._get_meta_dict

# # Apply the robust metadata extractor
# trainer._get_meta_dict = types.MethodType(robust_meta_dict, trainer)

# # Add batch inspection to the first iteration
# original_process_function = trainer.evaluator._process_function


# def debug_process_function(engine, batch):
#     inspect_batch(batch)
#     return original_process_function(engine, batch)


# trainer.evaluator._process_function = debug_process_function


# In[12]:


cfg.model.out_channels


# In[13]:


trainer.run()


# In[ ]:


# 9) Final memory report
if torch.cuda.is_available():
    peak_total = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[MEMORY] Peak GPU memory used across all: {peak_total:.2f} GB")
print_memory_stats("After trainer.run()")


# In[ ]:


import os
import nibabel as nib
from monai.transforms import LoadImage

# Path to the file
file_path = r"C:\Users\anson\Google Drive\00_CAREER AND EDUCATION\00_SGH Career\Prostate Cancer Project\prostate158-main\prostate158\train\121\t2.nii.gz"

print("=== Method 1: Using nibabel directly ===")
# Load with nibabel
nifti_img = nib.load(file_path)

# Get header info
header = nifti_img.header

# Print basic metadata
print(f"Image dimensions: {nifti_img.shape}")
print(f"Voxel dimensions (mm): {nifti_img.header.get_zooms()}")
print(f"Data type: {nifti_img.get_data_dtype()}")
print(f"Affine transform:\n{nifti_img.affine}")

# Print full header details
print("\nFull header details:")
print(header)

print("\n=== Method 2: Using MONAI LoadImage ===")
# Load with MONAI
loader = LoadImage(image_only=False)
img_data, meta_data = loader(file_path)

# Print the metadata
print("\nMONAI metadata:")
for key, value in meta_data.items():
    print(f"{key}: {value}")

# Show the actual keys that would be accessible in the transform pipeline
print("\nThe metadata key that would be used in transforms:")
print(
    f"'{os.path.basename(file_path).split('.')[0]}_meta_dict'"
)  # This should be 't2_meta_dict'


# Finish the training with final evaluation of the best model. To allow visualization of all outputs, add OutputStore handler first. Otherwise only output form the last epoch will be accessible.
# 

# In[15]:


eos_handler = ignite.handlers.EpochOutputStore()
eos_handler.attach(trainer.evaluator, "output")


# In[ ]:


torch.serialization.add_safe_globals([monai.data.meta_tensor.MetaTensor])
trainer.evaluate(checkpoint=r"models\network_tumor_3_key_metric=0.0000.pt")


# Generate a markdown document with segmentation results
# 

# In[ ]:


report_generator = ReportGenerator(cfg.run_id, cfg.out_dir, cfg.log_dir)
report_generator.generate_report()


# Have a look at some outputs
# 

# In[ ]:


output = trainer.evaluator.state.output
keys = ["image", "label", "pred"]
outputs = {k: [o[0][k].detach().cpu().squeeze() for o in output] for k in keys}


# In[ ]:


ListViewer(
    [o.transpose(0, 2).flip(-2) for o in outputs["image"][0:3]]
    + [o.argmax(0).transpose(0, 2).flip(-2).float() for o in outputs["label"][0:3]]
    + [o.argmax(0).transpose(0, 2).flip(-2).float() for o in outputs["pred"][0:3]]
).show()

