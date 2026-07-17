#!/usr/bin/env python3
"""Train TROGDOR on G1/G2/G3/G5, validate on G6.

Replaces the old train_bce.py / train_bce_PINTS.py / train_bce_SCREEN.py /
train_focaltversky.py / train_focal+tversky.py scripts, which differed only
in --loss and --tss_bed. BCE with --pos_weight is the current production
recipe (the shipped checkpoint); the other --loss choices are kept for
comparison but have not outperformed it in benchmarking so far.
"""

import argparse
import os

import torch

from chiaroscuro.data_transforms import normalization
from chiaroscuro.dataset import MixedBatchLoader, NascentDataset
from chiaroscuro.losses import focal_loss, focal_tversky_loss, tversky_loss
from chiaroscuro.trogdor import TROGDOR

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

TRAIN_SAMPLES = ["G1", "G2", "G3", "G5"]
VAL_SAMPLES = ["G6"]

NON_BCE_LOSSES = {
    "focal_tversky": focal_tversky_loss,
    "tversky": tversky_loss,
    "focal": focal_loss,
    "focal+tversky": lambda logits, y: focal_loss(logits, y) + tversky_loss(logits, y),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--loss",
    choices=["bce", *sorted(NON_BCE_LOSSES)],
    default="bce",
    help="Loss function. Default: bce, the current production recipe.",
)
parser.add_argument(
    "--pos_weight",
    type=float,
    default=500,
    help="Positive class weight for BCEWithLogitsLoss; only used when "
    "--loss=bce. Pass 0 to disable reweighting. Default: 500.",
)
parser.add_argument(
    "--tss_bed",
    default=os.path.join(DATA_DIR, "K562.positive.bed.gz"),
    help="TSS/TIR label BED, shared by all train and val samples. Default: "
    "K562.positive.bed.gz (dREG-derived; matches the shipped checkpoint). "
    "Other label sources tried: ENCSR220XSM_peaks.hg19.bed (PINTS), "
    "K562_ENCODE_prom_enh.hg19.bed (ENCODE SCREEN cCREs) — note these are "
    "different definitions of 'positive', not just different files, so "
    "--pos_weight and other defaults tuned against one may not transfer.",
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument(
    "--max_epochs", type=int, default=20, help="Maximum training epochs."
)
parser.add_argument(
    "--early_stopping",
    type=int,
    default=None,
    help="Stop after N epochs without validation-AUPRC improvement. Default: "
    "disabled, matching the current production recipe.",
)
parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
parser.add_argument(
    "--window_size", type=int, default=2**18, help="Training window size in bp."
)
parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
parser.add_argument(
    "--val_interval",
    type=int,
    default=None,
    help="Validate every N training steps in addition to epoch end.",
)
parser.add_argument(
    "--run_name", default=None, help="Override the auto-generated run/checkpoint name."
)
parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging.")
args = parser.parse_args()

os.makedirs(MODEL_DIR, exist_ok=True)

train_pl = [os.path.join(DATA_DIR, f"{s}.pl.bw") for s in TRAIN_SAMPLES]
train_mn = [os.path.join(DATA_DIR, f"{s}.mn.bw") for s in TRAIN_SAMPLES]
train_tss = [args.tss_bed] * len(TRAIN_SAMPLES)

val_pl = [os.path.join(DATA_DIR, f"{s}.pl.bw") for s in VAL_SAMPLES]
val_mn = [os.path.join(DATA_DIR, f"{s}.mn.bw") for s in VAL_SAMPLES]
val_tss = [args.tss_bed] * len(VAL_SAMPLES)

# TSS-centered: one window per annotated TSS (focused positives)
tss_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=normalization,
    rc_prob=0.5,
    max_jitter=args.window_size // 16,
    tss_centered=True,
    window_size=args.window_size,
)
# Genome-wide tiled: captures true negatives
tiled_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=normalization,
    rc_prob=0.5,
    max_jitter=args.window_size // 16,
    window_size=args.window_size,
)
# MixedBatchLoader: 7/8 TSS-centered + 1/8 tiled per batch
train_loader = MixedBatchLoader(
    tss_dataset,
    tiled_dataset,
    batch_size=args.batch_size,
    tss_fraction=7 / 8,
    num_workers=4,
    pin_memory=True,
)

# Validation dataset (G6, tiled, labels required for metrics)
val_dataset = NascentDataset(
    val_pl,
    val_mn,
    tss_beds=val_tss,
    transform=normalization,
    window_size=args.window_size,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# --- Loss function ---
if args.loss == "bce":
    loss_fn = (
        torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))
        if args.pos_weight
        else None  # TROGDOR default: unweighted BCEWithLogitsLoss
    )
    loss_tag = f"BCE_{args.pos_weight:g}" if args.pos_weight else "BCE"
else:
    loss_fn = NON_BCE_LOSSES[args.loss]
    loss_tag = args.loss

label_tag = os.path.basename(args.tss_bed).split(".")[0]
run_name = args.run_name or f"TROGDOR_{loss_tag}_{label_tag}_{args.lr:g}"

model = TROGDOR(name=os.path.join(MODEL_DIR, run_name), loss_fn=loss_fn).cuda()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)

# --- LR schedule: linear warmup + cosine decay ---
total_steps = args.max_epochs * len(train_loader)
decay_steps = max(1, total_steps - args.warmup_steps)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=args.warmup_steps
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=decay_steps, eta_min=1e-6
        ),
    ],
    milestones=[args.warmup_steps],
)

run = None
if not args.no_wandb:
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb is required unless --no_wandb is passed. "
            "Install dev dependencies with: pip install -e '.[dev]'"
        ) from e

    run = wandb.init(
        project="TROGDOR",
        name=run_name,
        config={
            "loss": args.loss,
            "pos_weight": args.pos_weight if args.loss == "bce" else None,
            "tss_bed": args.tss_bed,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "early_stopping": args.early_stopping,
            "warmup_steps": args.warmup_steps,
            "total_steps": total_steps,
            "window_size": args.window_size,
            "val_interval": args.val_interval,
        },
    )

model.fit(
    train_loader,
    optimizer,
    val_loader,
    max_epochs=args.max_epochs,
    batch_size=args.batch_size,
    early_stopping=args.early_stopping,
    verbose=True,
    wandb_run=run,
    scheduler=scheduler,
    val_interval=args.val_interval,
)

if run is not None:
    run.finish()
