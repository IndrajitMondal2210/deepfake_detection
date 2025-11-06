# train_fusion_transformer.py
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# ----------------------------
# Paths & Hyperparams
# ----------------------------
METADATA = "data/raw/meta_data.csv"
MODEL_OUT_DIR = "models/fusion_model"
MODEL_OUT = os.path.join(MODEL_OUT_DIR, "fusion_transformer.pth")
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

VISUAL_MODEL_PATH = "models/visual_model/visual_model.pth"
AUDIO_MODEL_PATH = "models/audio_model/audio_model.pth"
SYNC_MODEL_PATH  = "models/sync_model/sync_model.pth"

BATCH_SIZE = 16
EPOCHS = 6
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer hyperparams
D_MODEL = 64          # embedding dimension per token
N_HEADS = 8
NUM_LAYERS = 2
DIM_FF = 256
DROPOUT = 0.1

# ----------------------------
# Dataset (same as yours)
# ----------------------------
class FusionDataset(Dataset):
    def __init__(self, metadata_csv, feature_dir="data/processed/fusion_features"):
        self.df = pd.read_csv(metadata_csv)
        self.feature_dir = feature_dir
        self.samples = []
        # NOTE: keep same filter you had
        filtered_df = self.df[self.df['race'].isin(["Asian (East)", "Asian (South)"])]
        for idx, row in filtered_df.iterrows():
            video_id = os.path.splitext(row['path'])[0]
            label = 1 if row['category'] in ['C', 'D'] else 0
            visual_feat_path = os.path.join(feature_dir, f"visual_{video_id}.npy")
            audio_feat_path  = os.path.join(feature_dir, f"audio_{video_id}.npy")
            sync_feat_path   = os.path.join(feature_dir, f"sync_{video_id}.npy")
            if os.path.exists(visual_feat_path) and os.path.exists(audio_feat_path) and os.path.exists(sync_feat_path):
                self.samples.append((visual_feat_path, audio_feat_path, sync_feat_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vf, af, sf, label = self.samples[idx]
        visual = np.load(vf).astype(np.float32).ravel()
        audio  = np.load(af).astype(np.float32).ravel()
        sync   = np.load(sf).astype(np.float32).ravel()
        fusion_feat = np.concatenate([visual, audio, sync], axis=0)
        return fusion_feat, int(label)

# ----------------------------
# Transformer Fusion Model
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class FusionTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=8, num_layers=2, dim_feedforward=256, dropout=0.1, num_classes=2):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # compute sequence length by splitting input_dim into chunks of size d_model
        self.seq_len = math.ceil(input_dim / d_model)
        padded_dim = self.seq_len * d_model
        self.pad_len = padded_dim - input_dim  # number of zeros to pad

        # projection from flattened patches to d_model (here each token = d_model dims of original vector)
        # but since we already split into d_model-sized chunks, no linear required per chunk.
        # We'll reshape to (batch, seq_len, d_model) directly after padding.
        # To allow learnable projection, use a linear layer per token:
        self.token_proj = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.seq_len + 10)

        # classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, input_dim]
        batch = x.size(0)

        # pad if necessary
        if self.pad_len > 0:
            pad = x.new_zeros((batch, self.pad_len))
            x = torch.cat([x, pad], dim=1)  # [batch, seq_len * d_model]

        # reshape into tokens: [batch, seq_len, d_model]
        x = x.view(batch, self.seq_len, self.d_model)

        # optional linear projection per token
        x = self.token_proj(x)  # [batch, seq_len, d_model]

        # add positional encoding
        x = self.pos_enc(x)

        # transformer encoder
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # pooling: mean over seq dimension
        x = x.mean(dim=1)  # [batch, d_model]

        logits = self.classifier(x)  # [batch, num_classes]
        return logits

# ----------------------------
# Instantiate dataset / dataloader
# ----------------------------
dataset = FusionDataset(METADATA)
if len(dataset) == 0:
    raise RuntimeError("No samples found in FusionDataset. Check paths and fusion_features existence.")

# choose num_workers=0 on Windows to avoid spawn issues; use >0 on Linux
num_workers = 0 if os.name == 'nt' else 4
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

# ----------------------------
# Build model
# ----------------------------
sample_feat, _ = dataset[0]
input_dim = int(sample_feat.shape[0])
model = FusionTransformer(input_dim=input_dim, d_model=D_MODEL, n_heads=N_HEADS, num_layers=NUM_LAYERS, dim_feedforward=DIM_FF, dropout=DROPOUT).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)
    for feats, labels in loop:
        feats = feats.to(DEVICE)
        labels = labels.to(DEVICE).long()

        optimizer.zero_grad()
        outputs = model(feats)  # [B, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * feats.size(0)
        _, preds = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=f"{(correct/total):.3f}")

    epoch_loss = running_loss / len(dataset)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    scheduler.step()

# ----------------------------
# Save model
# ----------------------------
torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": input_dim,
    "d_model": D_MODEL,
    "seq_len": model.seq_len
}, MODEL_OUT)
print(f"✅ Saved fusion transformer model at {MODEL_OUT}")
