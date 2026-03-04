#!/usr/bin/env python3
"""Train TROGDOR on G1/G2/G3/G5, validate on G6, with early stopping."""

import os

import torch
import wandb

from burninate.dataset import MixedBatchLoader, NascentDataset
from burninate.trogdor import TROGDOR, standardization

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

# --- Training datasets ---
# TSS-centered: one window per annotated TSS (focused positives)
tss_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=standardization,
    rc_prob=0.5,
    max_jitter=2**14,
    tss_centered=True,
)
# Genome-wide tiled: captures true negatives
tiled_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=standardization,
    rc_prob=0.5,
    max_jitter=2**14,
)

# MixedBatchLoader: 7/8 TSS-centered + 1/8 tiled per batch
train_loader = MixedBatchLoader(
    tss_dataset,
    tiled_dataset,
    batch_size=64,
    tss_fraction=7 / 8,
    num_workers=4,
    pin_memory=True,
)

# --- Validation dataset (G6, tiled, labels required for metrics) ---
val_dataset = NascentDataset(
    val_pl,
    val_mn,
    tss_beds=val_tss,
    transform=standardization,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# --- Model + optimizer ---
model = TROGDOR(name="TROGDOR", pos_weight=10).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# --- Wandb ---
run = wandb.init(
    project="TROGDOR",
    config={
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 20,
        "early_stopping": 5,
    },
)

# --- Train ---
model.fit(
    train_loader,
    optimizer,
    val_loader,
    max_epochs=20,
    batch_size=64,
    early_stopping=5,
    verbose=True,
    wandb_run=run,
)
run.finish()
