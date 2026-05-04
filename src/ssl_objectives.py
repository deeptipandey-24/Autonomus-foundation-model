import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAutoencoderLoss(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, token_seq, tokens_in, mask):
        pred = self.decoder(token_seq)
        if mask.sum() == 0:
            return pred.sum() * 0.0
        return F.mse_loss(pred[mask], tokens_in[mask].detach())


class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, d_model=256, proj_dim=128):
        super().__init__()
        self.cam_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.lid_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, cam_emb, lid_emb):
        z_cam  = F.normalize(self.cam_proj(cam_emb), dim=-1)
        z_lid  = F.normalize(self.lid_proj(lid_emb), dim=-1)
        scale  = self.logit_scale.exp().clamp(max=100)
        logits = scale * z_cam @ z_lid.T
        labels = torch.arange(len(z_cam), device=z_cam.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class SSLObjective(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.mae         = MaskedAutoencoderLoss(d_model)
        self.contrastive = CrossModalContrastiveLoss(d_model)

    def forward(self, model_out, cam_emb, lid_emb):
        mae_loss  = self.mae(model_out['token_seq'],
                             model_out['tokens_in'],
                             model_out['mask'])
        cont_loss = self.contrastive(cam_emb, lid_emb)
        total     = mae_loss + cont_loss
        return {'loss': total, 'loss_mae': mae_loss.item(),
                'loss_contrastive': cont_loss.item()}


if __name__ == '__main__':
    ssl = SSLObjective()
    out = {
        'token_seq': torch.randn(2, 4, 256),
        'tokens_in': torch.randn(2, 4, 256),
        'mask':      torch.zeros(2, 4, dtype=torch.bool),
    }
    out['mask'][:, 1] = True
    losses = ssl(out, torch.randn(2, 256), torch.randn(2, 256))
    print('Total loss:      ', losses['loss'].item())
    print('MAE loss:        ', losses['loss_mae'])
    print('Contrastive loss:', losses['loss_contrastive'])