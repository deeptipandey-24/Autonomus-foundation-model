import os, sys, time, torch, torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset        import NuScenesMultimodalDataset
from fusion         import FusionTransformer
from ssl_objectives import SSLObjective
from task_heads     import AnomalyHead


def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        if k == 'token':
            out[k] = [b[k] for b in batch]
        else:
            t = torch.stack([b[k] for b in batch])
            out[k] = t.float() if t.is_floating_point() else t  # ← add this
    return out


def main():
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataroot = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'nuscenes'))
    print(f'Device: {device}')

    # Data
    train_ds = NuScenesMultimodalDataset(dataroot, split='train')
    val_ds   = NuScenesMultimodalDataset(dataroot, split='val')
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f'Train: {len(train_ds)} samples | Val: {len(val_ds)} samples')

    # Models
    model     = FusionTransformer(d_model=256).to(device)
    ssl_loss  = SSLObjective(d_model=256).to(device)
    anom_head = AnomalyHead(d_model=256).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer
    params    = list(model.parameters()) + list(ssl_loss.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5 * len(train_loader), eta_min=1e-6)

    os.makedirs('runs', exist_ok=True)
    best_val = float('inf')

    for epoch in range(1, 6):
        model.train(); ssl_loss.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch     = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            model_out = model(batch, mask_ratio=0.3)
            cam_emb   = model.cam_enc(batch['camera'])
            lid_emb   = model.lid_enc(batch['lidar'])
            losses    = ssl_loss(model_out, cam_emb, lid_emb)
            anom_loss = anom_head.loss(model_out['embedding'].detach())
            loss      = losses['loss'] + 0.1 * anom_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            print(f'  Ep{epoch} step{step+1}/{len(train_loader)} | '
                  f'loss={loss.item():.4f} '
                  f'mae={losses["loss_mae"]:.4f} '
                  f'cont={losses["loss_contrastive"]:.4f}')

        # Validation
        model.eval(); ssl_loss.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch     = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                model_out = model(batch, mask_ratio=0.3)
                cam_emb   = model.cam_enc(batch['camera'])
                lid_emb   = model.lid_enc(batch['lidar'])
                val_loss += ssl_loss(model_out, cam_emb, lid_emb)['loss'].item()

        val_loss /= max(len(val_loader), 1)
        print(f'\nEpoch {epoch}/5 | train={total_loss/len(train_loader):.4f} | val={val_loss:.4f} | time={time.time()-t0:.1f}s\n')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_loss': val_loss}, 'runs/best_model.pt')
            print(f'  ✅ Best model saved!')

    print('Training complete! Checkpoint saved to runs/best_model.pt')


if __name__ == '__main__':
    main()