import torch
import torch.nn as nn

class BEVfoundaltion(nn.Module):
    def __init__(self,embed_dim=256):
        super().__init__()
        self.cam_backbone = nn.Sequential(
            nn.conv2d(3,64,3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16,16))
        )

        self.lidar_backbone = nn.Sequential(
            nn.conv2d(3,64,3),
            nn.ReLU(),
        )
        self.transformer=nn.TransformerEncoder(
            nn.TransformerEncoder(d_model=embed_dim,nhead=8),
            num_layers=3
        )

    def forward(self,cam_img,lidar_img,maps_feat):
        fused_input=torch.cat([cam_img,lidar_img,maps_feat],dim=1)
        return self.transformer(fused_input)