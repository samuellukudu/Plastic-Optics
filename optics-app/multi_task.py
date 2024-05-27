import os, gc, sys
from pathlib import Path
import glob

import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import utils
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
    "encoder": "resnet34",
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
artifact_name = 'samu2505/PlasticOpticsMultiClassification/aerialMultiFPNModel_fold_0:v20'
artifact_type = 'model'

# Define the local directory to save the artifact
local_dir = f'{os.getcwd()}/multi-task-artifacts'

# Check if the artifact directory already exists locally
if not os.path.exists(local_dir):
    # If not, use the artifact and download it
    artifact = run.use_artifact(artifact_name, type=artifact_type)
    artifact_dir = artifact.download(root=local_dir)
else:
    # If it exists, print a message indicating it's already downloaded
    print(f"Artifact already downloaded in {local_dir}")


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


class MultiResnetFPN(nn.Module):
    def __init__(self, name, site_classes, severe_classes, pretrained=False):
        super(MultiResnetFPN, self).__init__()
        self.site_classes = site_classes
        self.severe_classes = severe_classes
        self.encoder = timm.create_model(name, pretrained=pretrained, features_only=True)
        
        # first backbone layers
        self.stage0 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.act1,
                                   self.encoder.maxpool)
        
        # Backbone layers (bottom-up layer)
        self.stage1 = nn.Sequential(self.encoder.layer1)
        self.stage2 = nn.Sequential(self.encoder.layer2)
        self.stage3 = nn.Sequential(self.encoder.layer3)
        self.stage4 = nn.Sequential(self.encoder.layer4)
        
        if 'resnet18' in name.lower() or 'resnet34' in name.lower():
            in_chans = self.encoder.layer4[-1].conv2.out_channels
        if 'resnet50' in name.lower():
            in_chans = self.encoder.layer4[-1].conv3.out_channels
        
        out_chans = in_chans // 8
        # Top Layer
        self.toplayer = nn.Conv2d(
            in_chans, out_chans, kernel_size=1, stride=1, padding=0)

        # Lateral Layers
        self.latlayer1 = nn.Conv2d(
            in_chans // 2, out_chans, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(
            in_chans // 4, out_chans, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(
            out_chans, out_chans, kernel_size=1, stride=1, padding=0)
        
        # smooth layers
        mid_chans = in_chans // 2 - out_chans
        self.smooth1 = nn.Conv2d(in_chans // 4, out_chans, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(mid_chans, out_chans, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(in_chans // 2, out_chans, kernel_size=3, stride=1, padding=1)
        
        # fully connected layer
        self.site_fc = nn.Linear(out_chans, site_classes)
        self.site_classifier = nn.Linear(4*site_classes, site_classes)
        
        self.severe_fc = nn.Linear(out_chans, severe_classes)
        self.severe_classifier = nn.Linear(4*severe_classes, severe_classes)
        
    def forward(self, x):
        # bottom-up pathway 
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2).detach()
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        
        # top-down pathway
        p5 = self.toplayer(c5)
        p4 = self._upsample_cat(p5, self.latlayer1(c4))
        p3 = self._upsample_cat(p4, self.latlayer2(c3))
        p2 = self._upsample_cat(p3, self.latlayer3(c2))
        
        # smoothing (de-aliasing effect)
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        
        # Global Average Pooling
        p5 = gap2d(p5, keepdims=True)
        p4 = gap2d(p4, keepdims=True)
        p3 = gap2d(p3, keepdims=True)
        p2 = gap2d(p2, keepdims=True)
        
        # Flattening
        p5 = p5.view(p5.size(0), -1)
        p4 = p4.view(p4.size(0), -1)
        p3 = p3.view(p3.size(0), -1)
        p2 = p2.view(p2.size(0), -1)
        
        # Fully connected layers
        site_out5 = F.relu(self.site_fc(p5))
        site_out4 = F.relu(self.site_fc(p4))
        site_out3 = F.relu(self.site_fc(p3))
        site_out2 = F.relu(self.site_fc(p2))
        
        severe_out5 = F.relu(self.severe_fc(p5))
        severe_out4 = F.relu(self.severe_fc(p4))
        severe_out3 = F.relu(self.severe_fc(p3))
        severe_out2 = F.relu(self.severe_fc(p2))
        
        # concatenate the predictions (classification results) of each of the pyramid features
        site_out = torch.cat([site_out5, site_out4, site_out3, site_out2], dim=1)
        site_out = self.site_classifier(site_out)
        
        severe_out = torch.cat([severe_out5, severe_out4, severe_out3, severe_out2], dim=1)
        severe_out = self.severe_classifier(severe_out)
        return site_out, severe_out
    
    def _upsample_cat(self, x, y):
        _, _, H, W = y.size()
        upsampled_x = F.interpolate(
            x, size=(H,W), mode="nearest"
        )
        return torch.cat([upsampled_x, y], dim=1)

@st.cache_resource
def load_multi_task_model(weight_path, name):
    model = MultiResnetFPN(name=name, site_classes=6, severe_classes=4, pretrained=False)

    weights = torch.load(weight_path, map_location=torch.device("cpu"))
    model.load_state_dict(weights)
    return model
