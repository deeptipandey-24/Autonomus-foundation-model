import torch
import torch.nn as nn


class CameraEncoder(nn.Module):
    def __init__(self, d_model=256, freeze_backbone=False):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
            vit_dim = self.backbone.embed_dim
        except Exception:
            self.backbone = _SimpleCNN()
            vit_dim = 512
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.proj = nn.Sequential(nn.LayerNorm(vit_dim), nn.Linear(vit_dim, d_model))

    def forward(self, x):
        return self.proj(self.backbone(x))


class _SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 512
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 512)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class LiDAREncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),    nn.GELU(),
            nn.Linear(64, 128),  nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
        )
        self.proj = nn.Linear(256, d_model)

    def forward(self, x):
        B, N, C = x.shape
        feat = self.mlp(x.reshape(B * N, C)).reshape(B, N, 256)
        return self.proj(feat.max(dim=1).values)


class EgoEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),  nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, d_model),
        )

    def forward(self, x):
        return self.net(x)


class CANEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),  nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, d_model),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    B, d = 2, 256
    print('Camera:', CameraEncoder(d)(torch.randn(B, 3, 224, 224)).shape)
    print('LiDAR: ', LiDAREncoder(d)(torch.randn(B, 4096, 4)).shape)
    print('Ego:   ', EgoEncoder(d)(torch.randn(B, 6)).shape)
    print('CAN:   ', CANEncoder(d)(torch.randn(B, 6)).shape)