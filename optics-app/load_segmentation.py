import os, gc, sys, yaml, json, copy
from pathlib import Path
from collections import defaultdict
import glob

import math
import random
import numpy as np

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import utils
import torchvision
from torchvision import transforms as T
import streamlit as st
# from dotenv import load_dotenv
import wandb
# load_dotenv()
# api_key = os.getenv("WANDB_API_KEY")
api_key = st.secrets["WANDB_API_KEY"]
wandb.login(key=api_key)

os.environ['WANDB_MODE'] = 'offline'

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

def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('seeding done!!!')


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

seeding(PARAMS['seed'])

# Initialize a W&B run
run = wandb.init()

# Define the artifact name and type
artifact_name = 'samu2505/AerialWasteSegmentation/aerialwaste_segmentation_weights:v2'
artifact_type = 'model'

# Define the local directory to save the artifact
local_dir = f'{os.getcwd()}/artifact'

# Check if the artifact directory already exists locally
if not os.path.exists(local_dir):
    # If not, use the artifact and download it
    artifact = run.use_artifact(artifact_name, type=artifact_type)
    artifact_dir = artifact.download(root=local_dir)
else:
    # If it exists, print a message indicating it's already downloaded
    print(f"Artifact already downloaded in {local_dir}")


def create_model(params):
    model = smp.DeepLabV3Plus(
        encoder_name=params['encoder'],
        encoder_weights=None,
        in_channels=params['in_channels'],
        classes=params['num_classes'],
        activation=None
    )
    return model.to(params['device']).half()


@st.cache_resource
def load_model_weights(weight_path):
    model = create_model(PARAMS)
    weights = torch.load(weight_path, map_location=torch.device("cpu"))
    model.load_state_dict(weights)
    return model
