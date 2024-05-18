import os, gc, sys, yaml, json, copy
from pathlib import Path
from collections import defaultdict
import glob
from tqdm.auto import tqdm

import math
import random
import numpy as np

import cv2
import PIL
import matplotlib.pyplot as plt

from load_segmentation import load_model_weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import utils
import torchvision
from torchvision import transforms as T

import streamlit as st

PARAMS = {
    "device": torch.device("cuda") if torch.cuda.is_available() else "cpu",
    "encoder": "resnext50_32x4d",
    "encoder_weights": "imagenet",
    "num_classes": 22,
    "in_channels": 3,
    "batch_size": 4,
    "num_workers": 0,
    "epochs": 100,
    "lr": 3e-4,
    "img_size": [512, 512],
    "seed": 2024,
}

# Continue with your program using the local artifact directory
seg_model_dir = f'{os.getcwd()}/artifact'
weights_path = glob.glob(f"{seg_model_dir}/*.pth")

model = load_model_weights(weight_path=weights_path[0])

tsfm = T.Compose([
    T.Resize(PARAMS['img_size']),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def inference_pipeline(img_path, transform=tsfm):
    image = PIL.Image.open(img_path)
    tensor_image = transform(image).unsqueeze(0)

    # Convert the image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    model.eval()
    with torch.no_grad():
        logits = model(tensor_image)

    activated = F.softmax(logits, dim=1)
    mask = torch.argmax(activated, dim=1)
    mask = mask.squeeze().detach().cpu().numpy()
    new_size = mask.shape
    image = image.resize(new_size)
    return image, mask


def visualize_predictions(image, mask):
    # Create a figure with a specific size
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Display the original image
    axs[0].imshow(image)
    axs[0].set_title("Image", fontsize=30)
    axs[0].axis("off")

    # Display the original image with the mask overlay
    axs[1].imshow(image)
    axs[1].imshow(mask, alpha=0.7)
    axs[1].set_title("Predicted Mask Overlay", fontsize=30)
    axs[1].axis("off")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Use Streamlit's function to display the plot
    st.pyplot(fig)


def main_loop():
    st.title("Plastic-optics Insight")
    st.subheader("This app allows you to locate landfills from satellite images")
    
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the mode", 
                                    ["Segmentation", "Classification", "Model Inspection"])

    image_file = st.file_uploader("upload your image", type=["jpg", "png", "jpeg"])
    if not image_file:
        return None
    
    if app_mode == "Segmentation":
        image, mask = inference_pipeline(img_path=image_file)
        visualize_predictions(image=image, mask=mask)
    elif app_mode == "Classification":
        st.sidebar.text("Classification results will be shown here.")
        # TODO: Implement classification functionality
    elif app_mode == "Model Inspection":
        st.sidebar.text("Model inspection with GradCAM will be shown here.")
        # TODO: Implement GradCAM functionality

if __name__ == "__main__":
    main_loop()
