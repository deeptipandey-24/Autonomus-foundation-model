import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from encoders import CameraEncoder, LiDAREncoder, EgoEncoder, CANEncoder

N_MODALITIES = 4

class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=4, ff_dim=512, dropout=0.1, freeze_cam=False):
        super().__init__()
        self.d_model  = d_model
        self.cam_enc  = CameraEncoder(d_model, freeze_backbone=freeze_cam)
        self.lid_enc  = LiDAREncoder(d_model)
        self.ego_enc  = EgoEncoder(d_model)
        self.can_enc  = CANEncoder(d_model)
        self.mod_embed = nn.Embedding(N_MODALITIES, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.mask_token  = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def encode_modalities(self, batch):
        tokens = torch.stack([
            self.cam_enc(batch['camera']),
            self.lid_enc(batch['lidar']),
            self.ego_enc(batch['ego']),
            self.can_enc(batch['can']),
        ], dim=1)
        mod_ids = torch.arange(N_MODALITIES, device=tokens.device)
        return tokens + self.mod_embed(mod_ids).unsqueeze(0)

    def forward(self, batch, mask_ratio=0.0):
        tokens = self.encode_modalities(batch)
        B, N, d = tokens.shape
        mask = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        if mask_ratio > 0.0:
            n_mask = max(1, int(N * mask_ratio))
            for b in range(B):
                idx = torch.randperm(N, device=tokens.device)[:n_mask]
                mask[b, idx] = True
                tokens[b, idx] = self.mask_token.squeeze(0)
        tokens_in = tokens.clone()
        out       = self.norm(self.transformer(tokens))
        return {'embedding': out.mean(dim=1), 'token_seq': out,
                'mask': mask, 'tokens_in': tokens_in}

    def freeze(self):
        for p in self.parameters(): p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters(): p.requires_grad_(True)


if __name__ == '__main__':
    model = FusionTransformer()
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    batch = {
        'camera': torch.randn(2, 3, 224, 224),
        'lidar':  torch.randn(2, 4096, 4),
        'ego':    torch.randn(2, 6),
        'can':    torch.randn(2, 6),
    }
    out = model(batch, mask_ratio=0.3)
    print('Embedding:', out['embedding'].shape)
    print('Token seq:', out['token_seq'].shape)