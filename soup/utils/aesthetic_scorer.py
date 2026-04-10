# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor

from PIL import Image
from torchvision import transforms
ASSETS_PATH = resources.files("assets")

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device
        clip_model_name = "openai/clip-vit-large-patch14"
        self.clip = CLIPModel.from_pretrained(clip_model_name).eval().to(device=self.device)  #, dtype=self.dtype)
        self.mlp = MLPDiff()
        state_dict = torch.load("./soup/utils/aesthetics_model/sac+logos+ava1-l14-linearMSE.pth", map_location=self.device)
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device)  #, dtype=self.dtype)
        self.mlp.eval()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        self.target_size = 224

    def __call__(self, images):
        # image range: [0, 1], shape: (bs, 3, h, w)
        #images = ((images / 2) + 0.5).clamp(0, 1) 
        im_pix = transforms.Resize(self.target_size)(images)
        im_pix = self.normalize(im_pix).to(self.dtype)
        with torch.no_grad():
            embed = self.clip.get_image_features(pixel_values=im_pix)
            embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).flatten()
