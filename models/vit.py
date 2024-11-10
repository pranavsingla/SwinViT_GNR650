import torch
import torch.nn as nn
from transformers import ViTModel

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        self.vit = ViTModel.from_pretrained(config.get('pretrained_model', 'google/vit-base-patch16-224'))

    def forward(self, x):
        return self.vit(x).last_hidden_state
