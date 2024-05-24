import os, gc
from pathlib import Path
import glob

import numpy as np

import cv2
import PIL
import matplotlib.pyplot as plt

from segmentation import load_seg_model_weights
from multi_task import load_multi_task_model

import torch
from torchvision import transforms as T

import streamlit as st

PARAMS = {
    "img_size": [512, 512],
    "seed": 2024,
    "multi_encoder": "resnet18",
}

SEVERITY = [1, 2, 0, 3]
SITES = ['Degraded area', 'Production site', 'Agricultural area/farm area', 'Other', 'Abandoned area', 'Non production building']

# load segmentation model weights 
seg_model_dir = f'{os.getcwd()}/segmentation_artifact'
seg_weight_path = glob.glob(f"{seg_model_dir}/*.pth")[0]
seg_model = load_seg_model_weights(weight_path=seg_weight_path)

# load multi-task segmentation model weights
multi_task_dir = f'{os.getcwd()}/multi-task-artifacts'
multi_task_path = glob.glob(f"{multi_task_dir}/*.pth")[0]
multi_model = load_multi_task_model(weight_path=multi_task_path, name=PARAMS["multi_encoder"])

tsfm = T.Compose([
    T.Resize(PARAMS['img_size']),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


@st.cache_resource
def multi_task_inference(img_path, transform=tsfm):
    image = PIL.Image.open(img_path)
    tensor_image = transform(image).unsqueeze(0)

    # Convert the image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    multi_model.eval()
    with torch.no_grad():
        site_logits, severe_logits = multi_model(tensor_image)
        
    return site_logits, severe_logits


def visualize_severe(image, severe_logits):
    probas = severe_logits.softmax(dim=-1).detach().numpy().squeeze()
    total = np.arange(len(SEVERITY))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Display the image
    ax[0].imshow(image)
    ax[0].axis('off')  # Turn off axis

    # Plot the probabilities bar chart
    ax[1].barh(total, probas, align='center')
    ax[1].set_yticks(total)
    ax[1].set_yticklabels(SEVERITY)
    ax[1].invert_yaxis()  # labels read top-to-bottom
    ax[1].set_xlabel('Probability')
    ax[1].set_title('Severity Prediction')

    plt.tight_layout()
    st.pyplot(fig)
    gc.collect()


def visualize_sites(image, site_logits):
    probas = site_logits.softmax(dim=-1).detach().numpy().squeeze()
    total = np.arange(len(SITES))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Display the image
    ax[0].imshow(image)
    ax[0].axis('off')  # Turn off axis

    # Plot the probabilities bar chart
    ax[1].barh(total, probas, align='center')
    ax[1].set_yticks(total)
    ax[1].set_yticklabels(SITES)
    ax[1].invert_yaxis()  # labels read top-to-bottom
    ax[1].set_xlabel('Probability')
    ax[1].set_title('Site Type predictions')

    plt.tight_layout()
    st.pyplot(fig)
    gc.collect()


@st.cache_resource
def seg_inference_pipeline(img_path, transform=tsfm):
    image = PIL.Image.open(img_path)
    tensor_image = transform(image).unsqueeze(0)

    # Convert the image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    seg_model.eval()
    with torch.no_grad():
        logits = seg_model(tensor_image)

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
    gc.collect()


def main_loop():
    st.title("Plastic-optics Insight")
    st.subheader("This app allows you to locate landfills from satellite images")

    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the mode", 
                                    ["Locate landfills", "Model Inspection"])

    image_file = st.file_uploader("upload your image", type=["jpg", "png", "jpeg"])
    if not image_file:
        return None
    

    if app_mode == "Locate landfills":
        image, mask = seg_inference_pipeline(img_path=image_file)
        
        st.subheader("Segmentation results")
        visualize_predictions(image=image, mask=mask)
        site_logits, severe_logits = multi_task_inference(img_path=image_file)
        
        st.subheader("Severity of landfills")
        visualize_severe(image=image, severe_logits=severe_logits)
        
        st.subheader("Site where landfills are located")
        visualize_sites(image=image, site_logits=site_logits)

    elif app_mode == "Model Inspection":
        st.sidebar.text("Model inspection with GradCAM will be shown here.")
        # TODO: Implement GradCAM functionality

    
    gc.collect()

if __name__ == "__main__":
    gc.collect()
    main_loop()
    