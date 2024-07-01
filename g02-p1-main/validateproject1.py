import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np

import torch
import timm
import argparse
import litdata
import matplotlib.pyplot as plt

from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import torchvision

from torch.optim import Adam, SGD

from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from src.main import LoRAWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    
    # Extract the model's state dict keys
    model_keys = set(model.state_dict().keys())

    for k, v in state_dict.items():
        if k in model_keys:  # Only keeping the keys that exist in both the model and the checkpoint
            new_state_dict[k] = v

    # Load the state dict into the model without strictness
    model.load_state_dict(new_state_dict, strict=False)

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer


def load_my_models():
    """
    Loads the fine-tuned models (full and LoRA).
    These should be saved in the output folder.
    Return the models in the order indicated below,
    so the teachers can test them.
    """
    model_path = "output/lora_model.pth"
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    for block in model.blocks:
        block.attn.qkv = LoRAWrapper(block.attn.qkv, rank=12)
        block.attn.proj = LoRAWrapper(block.attn.proj, rank=12)

    # Loading the model weights
    lora_model, _ = load_checkpoint(model_path, model)
    
    model_path =  "output/full_model.pth"
    full_model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)

    full_model, _ = load_checkpoint(model_path, model)
    return full_model, lora_model

def test_load_my_models():
    full_model, lora_model = load_my_models()
    # full_model = load_my_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = full_model.to(device)
    lora_model = lora_model.to(device)

    full_model.eval()
    lora_model.eval()

    # Send an example through the models, to check that they loaded properly
    test_img = torch.load('output/test_img.pth')
    with torch.no_grad():
        _ = full_model(test_img.unsqueeze(0).to(device))
        _ = lora_model(test_img.unsqueeze(0).to(device))

    print("Done!")

if __name__ == '__main__':
    test_load_my_models()

    