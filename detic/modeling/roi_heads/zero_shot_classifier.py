# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec

import os
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_similarity_heatmap(similarity_matrix, level_name, dataset_name, out_path, counter):
    # Convert similarity matrix to numpy for visualization
    similarity_matrix = similarity_matrix.detach().cpu().numpy()

    # Create the figure for the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='coolwarm', annot=False, fmt='.2f', cbar=True)

    # Set title and axis labels
    plt.title(f"Similarity Heatmap for {level_name} ({dataset_name})", fontsize=16)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Samples', fontsize=14)

    # Check if the base output directory exists, create if not
    if out_path:
        # Define the path for saving the heatmap image
        save_dir = os.path.join(out_path, dataset_name, level_name)
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Save the heatmap image with the counter value appended
        save_path = os.path.join(save_dir, f"{dataset_name}_{level_name}_similarity_heatmap_{counter}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved at {save_path}")  # Print the save path for confirmation

    # Tight layout and show
    plt.tight_layout()
    plt.show()


# 示例使用
level_name = 'l5'
dataset_name = 'iNat'
out_path = './heatmap/hm_image'


class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.path = zs_weight_path
        self.use_bias = use_bias < 0
        self.counter = 0  # 初始化计数器
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)
        
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                np.load(zs_weight_path), 
                dtype=torch.float32).permute(1, 0).contiguous() # D x C
        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros((zs_weight_dim, 1))], 
            dim=1) # D x (C + 1)
        
        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
        
        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape


    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
        }

    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous() # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight) # 是在这里传入npy? 这里报错了  zs_weight貌似就是npy的维度大小 這個就是來匹配計算相似度的？
        # visualize_similarity_heatmap(x, level_name, dataset_name, out_path, counter=self.counter)
        self.counter += 1  # 每调用一次 forward 增加一次计数
        if self.use_bias:
            x = x + self.cls_bias
        return x