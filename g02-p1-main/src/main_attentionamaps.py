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
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm


class ToRGBTensor:
    
    def __call__(self, img):
        return F.to_tensor(img).expand(3, -1, -1) # Expand to 3 channels
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
#Loads in data and returns the dataloader

 
class MyCustomAttentionMap(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
      
        self.num_heads = attention_module.num_heads
        self.head_dim = attention_module.head_dim
        self.scale = attention_module.scale
        self.qkv = attention_module.qkv
        self.q_norm = attention_module.q_norm
        self.k_norm = attention_module.k_norm
        self.attn_drop = attention_module.attn_drop
        self.proj = attention_module.proj
        self.proj_drop = attention_module.proj_drop
        self.attn_matrix = None 

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        self.attn_matrix = attn

        return x





def visualize_attention( img, patch_size, attentions):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    

    nh = attentions.shape[1]  # number of head

   
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].detach().numpy()

    return attentions


def plot_attention( attention):
    n_heads = attention.shape[0]

    plt.figure(figsize=(10, 10))
    image_numpy = image.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize the image
    image_numpy = image_numpy * np.array(in_std) + np.array(in_mean)
    image_numpy = np.clip(image_numpy, 0, 1)  # Clip values to [0, 1] range

    # Plot the image
    plt.imshow(image_numpy)
    #plt.imshow(np.mean(attention, 0), cmap='viridis', alpha=0.7)
    plt.imshow(np.mean(attention, 0), cmap='jet', alpha=0.7)
    plt.savefig("Attention map")
    plt.tight_layout()
    plt.show()

class MyCustomAttentionMapLoRA(nn.Module):
    def __init__(self, attention_module):
        super().__init__()

        # Basic parameters
        self.num_heads = attention_module.num_heads
        self.head_dim = attention_module.head_dim
        self.scale = attention_module.scale

        # Parameters for Q, K, V projections
        self.qkv = attention_module.qkv

        # Check if attention_module has LoRA components
        self.is_lora = hasattr(attention_module, 'qkv.orig_weight')

        # If LoRA model, fetch the additional parameters
        if self.is_lora:
            self.orig_weight = attention_module.qkv.orig_weight
            self.orig_bias = attention_module.qkv.orig_bias
            self.lora_linear = attention_module.qkv.lora_linear

        self.q_norm = attention_module.q_norm
        self.k_norm = attention_module.k_norm
        self.attn_drop = attention_module.attn_drop
        self.proj = attention_module.proj
        self.proj_drop = attention_module.proj_drop
        self.attn_matrix = None 

    def forward(self, x):
        B, N, C = x.shape

        if self.is_lora:
            # Use LoRA components for Q, K, V projections
            qkv = self.qkv(x)
            qkv += self.lora_linear[0](x)
            qkv = qkv + self.lora_linear[1](qkv)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        self.attn_matrix = attn

        return x




class LoRAWrapper(nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        assert isinstance(linear, nn.Linear)

        self.register_buffer('orig_weight', linear.weight.data.cpu())
        if linear.bias is not None:
            self.register_buffer('orig_bias', linear.bias.data.cpu())
        else:
            self.register_buffer('orig_bias', None)

        self.in_dim = linear.in_features
        self.out_dim = linear.out_features
        self.bias = linear.bias is not None
        self.rank = rank

        self.lora_linear = nn.Sequential(
            nn.Linear(self.in_dim, rank, bias=False),
            nn.Linear(rank, self.out_dim, bias=self.bias)
        ).to(device)

        nn.init.normal_(self.lora_linear[0].weight.data, mean=0, std=0.01)

        if self.bias:
            nn.init.zeros_(self.lora_linear[1].weight.data)
            nn.init.zeros_(self.lora_linear[1].bias.data)

    def forward(self, x):
        W_0 = torch.matmul(x.to(device), self.orig_weight.t().to(device))

        if self.bias:
            W_0 += self.orig_bias.to(device)

        A_x = self.lora_linear[0](x)
        BA_x = self.lora_linear[1](A_x)

        output = W_0 + BA_x

        return output
    

class MyCustomAttentionMap(nn.Module):
    def __init__(self, attention_module):
        super().__init__()

        # Basic parameters
        self.num_heads = attention_module.num_heads
        self.head_dim = attention_module.head_dim
        self.scale = attention_module.scale

        # Parameters for Q, K, V projections
        self.qkv = attention_module.qkv

        # Check if attention_module has LoRA components
        self.is_lora = hasattr(attention_module, 'qkv.orig_weight')

        # If LoRA model, fetch the additional parameters
        if self.is_lora:
            self.orig_weight = attention_module.qkv.orig_weight
            self.orig_bias = attention_module.qkv.orig_bias
            self.lora_linear = attention_module.qkv.lora_linear

        self.q_norm = attention_module.q_norm
        self.k_norm = attention_module.k_norm
        self.attn_drop = attention_module.attn_drop
        self.proj = attention_module.proj
        self.proj_drop = attention_module.proj_drop
        self.attn_matrix = None 

    def forward(self, x):
        B, N, C = x.shape

        if self.is_lora:
            # Use LoRA components for Q, K, V projections
            qkv = self.qkv(x)
            qkv += self.lora_linear[0](x)
            qkv = qkv + self.lora_linear[1](qkv)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        self.attn_matrix = attn

        return x



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


# Define command-line arguments to specify the model type (lora=True or lora=False)
parser = argparse.ArgumentParser(description="Load a ViT model with or without LoRA.")
parser.add_argument("--lora", action="store_true", help="Load the model with LoRA")
args = parser.parse_args()
 # Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Set seed (if required later)
seed = 1
batch_size = 128

# Define mean and std from ImageNet data
in_mean = [0.485, 0.456, 0.406]
in_std = [0.229, 0.224, 0.225]
    
datapath = '/projects/ec232/data/'

# Define postprocessing / transform of data modalities
postprocess = (
    T.Compose([                        
        ToRGBTensor(), 
        T.Resize((224,224), antialias=None),
        T.Normalize(in_mean, in_std),
    ]),
    nn.Identity(), 
)

traindata = litdata.LITDataset('ImageWoof', datapath).map_tuple(*postprocess)
train_dataloader = DataLoader(traindata, shuffle=True, batch_size=batch_size)

# Fetch one batch of images and labels
dataiter = iter(tqdm(train_dataloader, desc="Loading Data", leave=False))
images, labels = next(dataiter)

# Take the first image from the batch
image = images[0].to(device)
# Manually set the value for lora
if args.lora:

    # Define the path to load the model based on whether you're using LoRA or not
    model_path = "lora_model.pth"

   
    # Create the model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10).to(device)

    model.blocks[-1].attn = MyCustomAttentionMapLoRA(model.blocks[-1].attn)

    # Check if we need to modify the model for LoRA
    
    for block in model.blocks:
        block.attn.qkv = LoRAWrapper(block.attn.qkv, rank=12)
        block.attn.proj = LoRAWrapper(block.attn.proj, rank=12)

    # Loading the model weights
    model, _ = load_checkpoint(model_path, model)

    # Set the model to evaluation mode
    model.eval()

    # Pass the image through the model to get attention maps
    with torch.no_grad():
        output = model(image.unsqueeze(0))

    # Extract the attention matrix, visualize it, and plot
    attention_matrix = model.blocks[-1].attn.attn_matrix
    att = visualize_attention(image, 16, attention_matrix)
    plot_attention(att)

else:

    # Define the path to load the model based on whether we're using LoRA or not
    model_path =  "full_model.pth"

    # Load the selected model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10).to(device)

    # Replace the attention module in the model with custom one for visualization
    model.blocks[-1].attn = MyCustomAttentionMap(model.blocks[-1].attn)

    model, _ = load_checkpoint(model_path, model)


    # Set the model to evaluation mode
    model.eval()

    # Pass the image through the model to get attention maps
    with torch.no_grad():
        output = model(image.unsqueeze(0))

    # Extract the attention matrix, visualize it, and plot
    attention_matrix = model.blocks[-1].attn.attn_matrix
    att = visualize_attention(image, 16, attention_matrix)
    plot_attention(att)