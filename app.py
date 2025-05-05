import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import densenet121
import torch.nn as nn

# 👈 New additions for brain region detection
from nilearn import datasets, image, masking
from skimage.measure import label
import tempfile
import os

st.title("Alzheimer's MRI Classification & Region Detection")

# Load model
model = densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 4)
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
model.eval()

class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_tensor = transform(image_pil).unsqueeze(0)

    # Normalize image
    input_tensor = image_tensor
    image_np = np.array(image_pil.resize((224, 224))).astype(np.float32) / 255.0

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)
        st.subheader("Prediction:")
        st.success(f"🧠 The MRI scan is classified as: **{class_names[prediction.item()]}**")

    # Grad-CAM setup
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    targets = [ClassifierOutputTarget(prediction.item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    st.image(cam_image, caption="Grad-CAM Highlighted MRI", use_column_width=True)

    # 👈 New: Save temp heatmap for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        cam_path = os.path.join(tmpdir, "heatmap.png")
        cv2.imwrite(cam_path, (grayscale_cam * 255).astype(np.uint8))

        # Simulate region detection using threshold
        threshold = np.percentile(grayscale_cam, 95)
        region_mask = grayscale_cam > threshold
        region_mask_img = np.zeros_like(grayscale_cam)
        region_mask_img[region_mask] = 1

        # Use Harvard-Oxford atlas for identification
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = image.load_img(atlas.maps)
        atlas_labels = atlas.labels

        # Resize region_mask to atlas shape
        from skimage.transform import resize
        region_resized = resize(region_mask_img, atlas_img.shape, mode='constant')

        # Binarize
        region_bin = (region_resized > 0.5).astype(int)

        # Mask and extract region
        atlas_data = atlas_img.get_fdata()
        masked_data = atlas_data * region_bin
        region_idx = int(np.bincount(masked_data.astype(int).flatten()).argmax())
        region_name = atlas_labels[region_idx] if region_idx < len(atlas_labels) else "Unknown"

        st.subheader("🧠 Anatomical Focus Region:")
        st.success(f"📍 Most focused brain region: **{region_name}**")

        # Optional explanations
        region_explanations = {
            "Temporal Pole": "Associated with early memory loss.",
            "Hippocampus": "Critical for memory — commonly affected in Alzheimer's.",
            "Frontal Pole": "Linked with planning and decision-making, impacted in later stages.",
        }
        if region_name in region_explanations:
            st.info(region_explanations[region_name])
        else:
            st.warning("Explanation not available for this region.")
