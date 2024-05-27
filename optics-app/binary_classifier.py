import os, gc, sys
from pathlib import Path
import glob

import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

import timm
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
    "encoder": "efficientnet_b0.ra_in1k",
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
artifact_name = 'samu2505/PlasticOpticsBinaryClassification/aerialMultiFPNModel_fold_0:v11'
artifact_type = 'model'

# Define the local directory to save the artifact
local_dir = f'{os.getcwd()}/binary-artifacts'

# Check if the artifact directory already exists locally
if not os.path.exists(local_dir):
    # If not, use the artifact and download it
    artifact = run.use_artifact(artifact_name, type=artifact_type)
    artifact_dir = artifact.download(root=local_dir)
else:
    # If it exists, print a message indicating it's already downloaded
    print(f"Artifact already downloaded in {local_dir}")


class AerialModel(nn.Module):
    def __init__(self, 
                 name: str, 
                 num_classes: int = 1, 
                 pretrained: bool = False, 
                 kernel_size: int = 3, 
                 stride: int = 2):
        
        super().__init__()
        self.encoder = timm.create_model(name, pretrained=pretrained, num_classes=0)
        nb_fts = self.encoder.num_features
        nb_fts = nb_fts // stride
        self.nb_fts = nb_fts if kernel_size < 3 else nb_fts - 1
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=stride)
        
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.nb_fts, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.avg_pool(x)
        
        outputs = self.head(x)
        return outputs


@st.cache_resource
def load_binary_model(weight_path, name):
    model = AerialModel(name=name, pretrained=False)

    weights = torch.load(weight_path, map_location=torch.device("cpu"))
    model.load_state_dict(weights)
    return model