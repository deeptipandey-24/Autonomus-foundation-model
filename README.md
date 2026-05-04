# Autonomous Foundation Model

A multimodal self-supervised foundation model trained on vehicle logs from the nuScenes dataset. The model learns general-purpose embeddings from camera, LiDAR, ego-motion, and CAN bus data — applicable to downstream tasks including perception, trajectory prediction, and anomaly detection.

---

## Model Architecture

### Overview

The system follows an **encode → fuse → learn** pipeline:

```
Camera (RGB)     →  CameraEncoder   ─┐
LiDAR (Points)   →  LiDAREncoder    ─┤→ FusionTransformer → Embeddings
Ego Pose/Vel     →  EgoEncoder      ─┤
CAN Bus Signals  →  CANEncoder      ─┘
```

### Modality Encoders (`src/encoders.py`)

| Encoder | Input | Architecture | Output |
|---|---|---|---|
| CameraEncoder | (B, 3, 224, 224) | ViT-Small/16 (timm) + Linear proj | (B, 256) |
| LiDAREncoder | (B, 4096, 4) | Point-wise MLP + Max Pool | (B, 256) |
| EgoEncoder | (B, 6) | 3-layer MLP (pos + RPY) | (B, 256) |
| CANEncoder | (B, 6) | 3-layer MLP (velocity signals) | (B, 256) |

- **CameraEncoder** uses a pretrained ViT-Small backbone (optionally frozen for transfer learning), projecting to d_model via LayerNorm + Linear.
- **LiDAREncoder** processes raw point clouds (x, y, z, intensity) with a shared MLP across all points, then aggregates with max-pooling.
- **EgoEncoder / CANEncoder** encode 6-DoF pose/velocity signals via lightweight MLPs.

### Fusion Transformer (`src/fusion.py`)

The four modality tokens (one per encoder) are fused using a **Transformer Encoder** with modality-specific positional embeddings:

- 4 input tokens (one per modality) + modality embeddings
- 4 Transformer layers, 4 attention heads, d_model=256, ff_dim=512
- `norm_first=True` (Pre-LN) for training stability
- **Masked token modelling**: a configurable fraction of tokens are replaced with a learnable `[MASK]` token during training
- Output: global `embedding` (mean-pooled) + full `token_seq` for reconstruction

**Parameters: ~2.6M** (lightweight, suitable for CPU and edge deployment)

---

## Self-Supervised Learning Objectives (`src/ssl_objectives.py`)

Two complementary SSL objectives are used simultaneously:

### 1. Masked Autoencoder (MAE)
- Random modality tokens are masked (default 30%)
- A decoder reconstructs the original token embeddings from masked positions
- Loss: MSE between predicted and original token embeddings
- Encourages the model to learn cross-modal context and redundancy

### 2. Cross-Modal Contrastive Learning
- Camera and LiDAR embeddings are projected into a shared 128-dim space
- InfoNCE / CLIP-style contrastive loss aligns camera and LiDAR views of the same scene
- Pushes embeddings from different scenes apart
- Learnable temperature parameter (`logit_scale`)

**Total loss = MAE loss + Contrastive loss**

---

## Dataset & Data Processing (`src/dataset.py`)

- **Dataset**: nuScenes v1.0-mini (10 scenes, ~404 samples)
- **Split**: 80% train (8 scenes), 20% val (2 scenes)
- **Modalities loaded per sample**:
  - `CAM_FRONT`: resized to 224×224, ImageNet-normalized
  - `LIDAR_TOP`: 4096 points via Farthest Point Sampling (FPS)
  - `ego`: 6-DoF position + roll/pitch/yaw from ego pose
  - `can`: velocity approximated from consecutive ego poses (Δpose/Δtime)

**Custom `NuScenesNoMap`** subclass skips map table loading (not required for sensor data access).

---

## Training Pipeline (`train.py`)

```bash
python train.py
```

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR over 5 epochs
- **Gradient clipping**: max norm 1.0
- **Batch size**: 4 (drop_last=True)
- **Epochs**: 5
- **Device**: CUDA if available, else CPU
- **Checkpointing**: best model saved to `runs/best_model.pt`

Training logs per step:
```
Ep1 step1/81 | loss=1.4584 mae=0.0285 cont=1.3864
```

---

## Downstream Task Heads (`src/task_heads.py`)

Three plug-and-play task heads operate on the frozen or fine-tuned backbone embedding:

| Head | Task | Architecture | Loss |
|---|---|---|---|
| `PerceptionHead` | Object classification (23 classes) | 2-layer MLP | CrossEntropy |
| `TrajectoryHead` | 12-step future trajectory | 3-layer MLP → (12, 2) | L1 |
| `AnomalyHead` | Reconstruction-based anomaly detection | Autoencoder bottleneck | MSE |

---

## Fine-Tuning & Transfer Learning

### Frozen Backbone
```python
model = FusionTransformer(d_model=256, freeze_cam=True)
model.freeze()  # freeze entire backbone
# attach task head and train only the head
```

### Full Fine-Tuning
```python
model.unfreeze()
# train end-to-end on downstream task
```

### Partial Fine-Tuning
```python
# Freeze only camera backbone, train the rest
model = FusionTransformer(freeze_cam=True)
```

---

## Continual Learning

### Elastic Weight Consolidation (EWC)
Implemented in `src/task_heads.py`:

```python
from task_heads import EWC

# After training task 1:
ewc = EWC(model, task1_dataloader, device)

# While training task 2:
loss = task2_loss + ewc.penalty(model, importance=1000.0)
```

EWC computes the Fisher Information Matrix over task-1 data to identify important parameters, then penalizes large deviations from task-1 weights during task-2 training — preventing catastrophic forgetting.

---

## Project Structure

```
autonomous-foundation-model/
├── src/
│   ├── dataset.py          # NuScenes multimodal data loader
│   ├── encoders.py         # Camera, LiDAR, Ego, CAN encoders
│   ├── fusion.py           # FusionTransformer (cross-modal attention)
│   ├── ssl_objectives.py   # MAE + Contrastive SSL losses
│   └── task_heads.py       # Perception, Trajectory, Anomaly, EWC
├── train.py                # Main training script
├── check_dataset.py        # Dataset sanity check
├── model.py                # Model entry point
├── data/nuscenes/          # nuScenes v1.0-mini (not tracked)
├── runs/                   # Saved checkpoints (not tracked)
└── README.md
```

---

## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/deeptipandey-24/autonomous-foundation-model.git
cd autonomous-foundation-model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision
pip install nuscenes-devkit
pip install timm
pip install pyquaternion
pip install Pillow numpy
```

**Download nuScenes data:**
1. Register at https://www.nuscenes.org/
2. Download `v1.0-mini` and map expansion pack v1.3
3. Place under `data/nuscenes/`

---

## Results (nuScenes v1.0-mini)

| Metric | Epoch 1 | Epoch 5 |
|---|---|---|
| Train Loss | 1.458 | converging |
| MAE Loss | 0.028 → 0.001 | ~0.001 |
| Contrastive Loss | 1.386 | decreasing |

The MAE loss drops rapidly (28x reduction in epoch 1), indicating the model quickly learns cross-modal reconstruction. The contrastive loss decreases more gradually as camera-LiDAR alignment is a harder objective.

---

## Design Decisions

- **ViT over CNN for camera**: better global context, transferable pretrained weights
- **Max-pool over attention for LiDAR**: efficient for large point clouds, permutation invariant
- **4 Transformer layers**: balance between capacity and CPU trainability
- **MAE + Contrastive**: complementary objectives — MAE captures intra-scene structure, contrastive captures cross-modal alignment
- **EWC for continual learning**: parameter-space regularization, no replay buffer needed
- **Modality embeddings**: allow the transformer to distinguish token sources without positional encoding

---

## Future Work

- Add radar and GPS modalities
- Scale to full nuScenes dataset (700 scenes)
- Replace velocity-approximated CAN with real CAN bus signals
- Add BEV (Bird's Eye View) projection head
- Evaluate embeddings on nuScenes detection/tracking benchmarks
