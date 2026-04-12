#!/usr/bin/env python3
"""Train TROGDOR on G1/G2/G3/G5, validate on G6, with early stopping."""

import argparse
import os

import torch
import wandb

from chiaroscuro.data_transforms import normalization
from chiaroscuro.dataset import MixedBatchLoader, NascentDataset
from chiaroscuro.trogdor import TROGDOR

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pos_weight",
    type=float,
    default=500,
    help="Positive class weight for BCEWithLogitsLoss. Upweights positive bins "
    "to compensate for class imbalance. Default: 500.",
)
parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate. Default: 1e-3."
)
args = parser.parse_args()

TRAIN_SAMPLES = ["G1", "G2", "G3", "G5"]
VAL_SAMPLES = ["G6"]
TSS_BED = os.path.join(DATA_DIR, "K562_ENCODE_prom_enh.hg19.bed")

train_pl = [os.path.join(DATA_DIR, f"{s}.pl.bw") for s in TRAIN_SAMPLES]
train_mn = [os.path.join(DATA_DIR, f"{s}.mn.bw") for s in TRAIN_SAMPLES]
train_tss = [TSS_BED] * len(TRAIN_SAMPLES)

val_pl = [os.path.join(DATA_DIR, f"{s}.pl.bw") for s in VAL_SAMPLES]
val_mn = [os.path.join(DATA_DIR, f"{s}.mn.bw") for s in VAL_SAMPLES]
val_tss = [TSS_BED] * len(VAL_SAMPLES)

# --- Hyperparameters ---
MAX_EPOCHS = 20
WARMUP_STEPS = 500
BATCH_SIZE = 64  # halved from 64 to keep GPU memory constant with 2× window
EARLY_STOPPING = 5
WEIGHT_DECAY = 1e-4
WINDOW_SIZE = 2**18  # 524288 bp (2× the original 2^18)

# --- Training datasets ---
# TSS-centered: one window per annotated TSS (focused positives)
tss_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=normalization,
    rc_prob=0.5,
    max_jitter=2**15,  # scaled with window (was 2**14)
    tss_centered=True,
    window_size=WINDOW_SIZE,
)
# Genome-wide tiled: captures true negatives
tiled_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=normalization,
    rc_prob=0.5,
    max_jitter=2**14,  # scaled with window (was 2**14)
    stride=2**17,  # 50% overlap (was 2**17 default)
    window_size=WINDOW_SIZE,
)

# MixedBatchLoader: 7/8 TSS-centered + 1/8 tiled per batch
train_loader = MixedBatchLoader(
    tss_dataset,
    tiled_dataset,
    batch_size=BATCH_SIZE,
    tss_fraction=7 / 8,
    num_workers=4,
    pin_memory=True,
)

# --- Validation dataset (G6, tiled, labels required for metrics) ---
val_dataset = NascentDataset(
    val_pl, val_mn, tss_beds=val_tss, transform=normalization, window_size=WINDOW_SIZE
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

# --- Loss function ---
if args.pos_weight is not None:
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))
else:
    loss_fn = None  # use TROGDOR default (BCEWithLogitsLoss with no reweighting)

# --- Model + optimizer ---

model = TROGDOR(
    name=f"{MODEL_DIR}/TROGDOR_BCE_SCREEN_{args.pos_weight}_{args.lr}", loss_fn=loss_fn
).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

# --- LR schedule: linear warmup + cosine decay ---
total_steps = MAX_EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=WARMUP_STEPS
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - WARMUP_STEPS, eta_min=1e-6
        ),
    ],
    milestones=[WARMUP_STEPS],
)

# --- Wandb ---
run = wandb.init(
    project="TROGDOR",
    config={
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
        "batch_size": BATCH_SIZE,
        "lr": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": EARLY_STOPPING,
        "warmup_steps": WARMUP_STEPS,
        "total_steps": total_steps,
        "loss_fn": "BCEWithLogitsLoss",
        "pos_weight": args.pos_weight,
        "window_size": WINDOW_SIZE,
    },
)

# --- Train ---
model.fit(
    train_loader,
    optimizer,
    val_loader,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping=EARLY_STOPPING,
    verbose=True,
    wandb_run=run,
    scheduler=scheduler,
)
run.finish()
