import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionHead(nn.Module):
    def __init__(self, d_model=256, n_classes=23):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, emb):
        return self.head(emb)

    def loss(self, emb, labels):
        return F.cross_entropy(self(emb), labels)


class TrajectoryHead(nn.Module):
    def __init__(self, d_model=256, horizon=12):
        super().__init__()
        self.horizon = horizon
        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, 128),     nn.GELU(),
            nn.Linear(128, horizon * 2),
        )

    def forward(self, emb):
        return self.head(emb).view(-1, self.horizon, 2)

    def loss(self, emb, gt):
        return F.l1_loss(self(emb), gt)


class AnomalyHead(nn.Module):
    def __init__(self, d_model=256, bottleneck=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128), nn.GELU(),
            nn.Linear(128, d_model),
        )

    def forward(self, emb):
        return self.decoder(self.encoder(emb))

    def anomaly_score(self, emb):
        return F.mse_loss(self(emb), emb, reduction='none').mean(dim=-1)

    def loss(self, emb):
        return F.mse_loss(self(emb), emb)


class EWC:
    """Elastic Weight Consolidation — prevents forgetting task 1 while training task 2."""
    def __init__(self, model, dataloader, device, n_samples=100):
        self._means  = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self._fisher = self._compute_fisher(model, dataloader, device, n_samples)

    @staticmethod
    def _compute_fisher(model, dataloader, device, n_samples):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model.zero_grad()
            out = model(batch)
            out['embedding'].pow(2).mean().backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
            count += len(out['embedding'])
        for n in fisher:
            fisher[n] /= max(count, 1)
        model.train()
        return fisher

    def penalty(self, model, importance=1000.0):
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self._fisher and p.requires_grad:
                loss = loss + (self._fisher[n] * (p - self._means[n]).pow(2)).sum()
        return importance * loss


if __name__ == '__main__':
    emb = torch.randn(4, 256)
    print('Perception:', PerceptionHead()(emb).shape)
    print('Trajectory:', TrajectoryHead()(emb).shape)
    print('Anomaly scores:', AnomalyHead().anomaly_score(emb))