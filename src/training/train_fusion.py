# src/training/train_fusion.py
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================
# Paths & Hyperparameters
# ============================
METADATA = "data/raw/meta_data.csv"
FUSION_FEAT_DIR = "data/processed/fusion_features"

MODEL_OUT_DIR = "models/fusion_model"
MODEL_OUT = os.path.join(MODEL_OUT_DIR, "fusion_transformer.pth")
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 6
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer hyperparams
D_MODEL = 64          # token/embedding size
N_HEADS = 8
NUM_LAYERS = 2
DIM_FF = 256
DROPOUT = 0.1
NUM_CLASSES = 4       # A, B, C, D

# ============================
# Dataset
# ============================
class FusionDataset(Dataset):
    """
    Loads concatenated fusion features (visual + audio + sync) per video.
    Expects files:
      data/processed/fusion_features/
        visual_<video_id>.npy
        audio_<video_id>.npy
        sync_<video_id>.npy
    Labels are mapped from metadata.csv 'category' as:
      A->0, B->1, C->2, D->3
    """
    def init(self, metadata_csv: str, feature_dir: str = FUSION_FEAT_DIR, race_filter=None):
        self.df = pd.read_csv(metadata_csv)
        if race_filter is not None:
            self.df = self.df[self.df["race"].isin(race_filter)]
        self.feature_dir = feature_dir
        self.samples = []
        self.cat2id = {"A": 0, "B": 1, "C": 2, "D": 3}

        for _, row in self.df.iterrows():
            # row['path'] looks like: FakeVideo-RealAudio/Black/men/id00166
            video_id = os.path.splitext(row["path"])[0]  # keep dir-like id; npy names use this suffix
            v = os.path.join(feature_dir, f"visual_{video_id}.npy")
            a = os.path.join(feature_dir, f"audio_{video_id}.npy")
            s = os.path.join(feature_dir, f"sync_{video_id}.npy")
            if all(os.path.exists(p) for p in [v, a, s]):
                label = self.cat2id.get(str(row["category"]).strip(), None)
                if label is not None:
                    self.samples.append((v, a, s, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                "FusionDataset is empty. Check that fusion feature .npy files exist and metadata.csv has matching paths."
            )

    def len(self):
        return len(self.samples)

    def getitem(self, idx):
        vf, af, sf, label = self.samples[idx]
        visual = np.load(vf).astype(np.float32).ravel()
        audio  = np.load(af).astype(np.float32).ravel()
        sync   = np.load(sf).astype(np.float32).ravel()
        feat = np.concatenate([visual, audio, sync], axis=0)
        return torch.from_numpy(feat), int(label)

# ============================
# Model: Transformer Fusion
# ============================
class PositionalEncoding(nn.Module):
    def init(self, d_model: int, max_len: int = 5000):
        super().init()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        return x + self.pe[:, : x.size(1), :]

class FusionTransformer(nn.Module):
    """
    Treat the 1D fusion vector as a sequence by splitting into tokens of size d_model.
    """
    def init(self, input_dim: int, num_classes: int = NUM_CLASSES,
d_model: int = D_MODEL, n_heads: int = N_HEADS, num_layers: int = NUM_LAYERS,
                 dim_ff: int = DIM_FF, dropout: float = DROPOUT):
        super().init()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        # sequence length and padding to align with d_model
        self.seq_len = math.ceil(input_dim / d_model)
        padded_dim = self.seq_len * d_model
        self.pad_len = padded_dim - input_dim

        # token projection (optional but helpful)
        self.token_proj = nn.Linear(d_model, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.seq_len + 8)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        B = x.size(0)
        if self.pad_len > 0:
            pad = x.new_zeros((B, self.pad_len))
            x = torch.cat([x, pad], dim=1)

        # [B, S, D]
        x = x.view(B, self.seq_len, self.d_model)
        x = self.token_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # mean pool over sequence
        return self.head(x)  # [B, NUM_CLASSES]

# ============================
# Train
# ============================
def train():
    # If you want to filter specific races, pass a list. Else None for all.
    dataset = FusionDataset(METADATA, FUSION_FEAT_DIR, race_filter=None)
    num_workers = 0 if os.name == "nt" else 4
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    # infer input dimension from a sample
    sample_x, _ = dataset[0]
    input_dim = int(sample_x.shape[0])

    model = FusionTransformer(input_dim=input_dim, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)

        for feats, labels in loop:
            feats = feats.to(DEVICE).float()
            labels = labels.to(DEVICE).long()

            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct/total):.3f}")

        epoch_loss = running_loss / len(dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
        scheduler.step()

    # save checkpoint with metadata for easy reload
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "d_model": D_MODEL,
        "seq_len": model.seq_len,
        "num_classes": NUM_CLASSES
    }, MODEL_OUT)
    print(f"✅ Saved fusion transformer model at {MODEL_OUT}")

if name == "main":
    train()