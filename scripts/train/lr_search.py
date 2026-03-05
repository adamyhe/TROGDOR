#!/usr/bin/env python3
"""LR hyperparameter search: train TROGDOR for 1000 steps across a log-spaced
grid of learning rates (1e-6 to 1e-3) and log results to wandb."""

import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torcheval.metrics.functional import binary_auprc

from burninate.dataset import MixedBatchLoader, NascentDataset
from burninate.predict import predict
from burninate.trogdor import TROGDOR, normalization

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

TRAIN_SAMPLES = ["G1", "G2", "G3", "G5"]
VAL_SAMPLES = ["G6"]
TSS_BED = os.path.join(DATA_DIR, "K562.positive.bed.gz")

train_pl = [os.path.join(DATA_DIR, f"{s}.pl.bw") for s in TRAIN_SAMPLES]
train_mn = [os.path.join(DATA_DIR, f"{s}.mn.bw") for s in TRAIN_SAMPLES]
train_tss = [TSS_BED] * len(TRAIN_SAMPLES)

val_pl = [os.path.join(DATA_DIR, f"{s}.pl.bw") for s in VAL_SAMPLES]
val_mn = [os.path.join(DATA_DIR, f"{s}.mn.bw") for s in VAL_SAMPLES]
val_tss = [TSS_BED] * len(VAL_SAMPLES)

# --- Datasets (built once, reused across all LRs) ---
tss_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=normalization,
    rc_prob=0.5,
    max_jitter=2**14,
    tss_centered=True,
)
tiled_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=normalization,
    rc_prob=0.5,
    max_jitter=2**14,
)
train_loader = MixedBatchLoader(
    tss_dataset,
    tiled_dataset,
    batch_size=64,
    tss_fraction=7 / 8,
    num_workers=4,
    pin_memory=True,
)

val_dataset = NascentDataset(
    val_pl,
    val_mn,
    tss_beds=val_tss,
    transform=normalization,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# --- LR search ---
LRS = np.logspace(-6, -3, num=7)

for lr in LRS:
    model = TROGDOR(name="TROGDOR", pos_weight=10).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    run = wandb.init(
        project="TROGDOR",
        name=f"lr_{lr:.2e}",
        group="lr_search",
        config={
            "lr": lr,
            "weight_decay": 1e-4,
            "train_steps": 1000,
            "batch_size": 64,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
        },
    )

    model.train()
    train_loss = 0.0
    for step, (X, y) in enumerate(train_loader):
        if step >= 1000:
            break
        X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(X)
            loss = model.loss(logits, y).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (step + 1) % 100 == 0:
            run.log({"train/bce": train_loss / (step + 1)}, step=step + 1)

    model.eval()
    with torch.no_grad():
        y_vals, logits_vals = [], []
        for X_val, y_val in val_loader:
            logits_vals.append(
                predict(
                    model, X_val, batch_size=64, device="cuda", dtype=torch.bfloat16
                )
            )
            y_vals.append(y_val)
        y_val_cat = torch.cat(y_vals)
        logits_val_cat = torch.cat(logits_vals)
        val_bce = F.binary_cross_entropy_with_logits(logits_val_cat, y_val_cat).item()
        val_auprc = binary_auprc(
            torch.sigmoid(logits_val_cat.reshape(-1)),
            y_val_cat.reshape(-1).long(),
        ).item()

    run.log({"val/bce": val_bce, "val/auprc": val_auprc}, step=1000)
    print(f"lr={lr:.2e}  val_bce={val_bce:.4f}  val_auprc={val_auprc:.4f}")
    run.finish()
