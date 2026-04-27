#!/usr/bin/env python3
"""Train the experimental TROGDORResidual architecture with BCE loss."""

import argparse
import os

import torch

from chiaroscuro.data_transforms import normalization
from chiaroscuro.dataset import MixedBatchLoader, NascentDataset
from chiaroscuro.trogdor import TROGDORResidual

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def parse_dilations(value):
    if value.lower() in {"", "none", "off"}:
        return ()
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pos_weight",
    type=float,
    default=500,
    help="Positive class weight for BCEWithLogitsLoss. Default: 500.",
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument(
    "--activation",
    choices=["silu", "gelu", "relu"],
    default="silu",
    help="Activation used in residual blocks. Default: silu.",
)
parser.add_argument(
    "--bottleneck_dilations",
    type=parse_dilations,
    default=(1, 2, 4, 8),
    help="Comma-separated bottleneck dilations, or 'none'. Default: 1,2,4,8.",
)
parser.add_argument(
    "--use_strand_features",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Add plus+minus and plus-minus channels inside the model. Default: true.",
)
parser.add_argument(
    "--output_stride",
    type=int,
    default=16,
    help="Output bin size in bp. Default: 16.",
)
parser.add_argument(
    "--base_channels",
    type=int,
    default=32,
    help="Base model width. Default: 32.",
)
parser.add_argument(
    "--context_depth",
    type=int,
    default=4,
    help="Number of retained-skip encoder/decoder levels. Default: 4.",
)
parser.add_argument(
    "--max_channels",
    type=int,
    default=512,
    help="Maximum channel count. Default: 512.",
)
parser.add_argument(
    "--window_size",
    type=int,
    default=2**18,
    help="Training window size in bp. Default: 262144.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Training batch size. Default: 64.",
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=20,
    help="Maximum training epochs. Default: 20.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=500,
    help="Linear warmup steps. Default: 500.",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
    help="AdamW weight decay. Default: 1e-4.",
)
parser.add_argument(
    "--val_interval",
    type=int,
    default=None,
    help="Validate every N training steps in addition to epoch end.",
)
parser.add_argument(
    "--no_wandb",
    action="store_true",
    help="Disable Weights & Biases logging.",
)
args = parser.parse_args()

os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_SAMPLES = ["G1", "G2", "G3", "G5"]
VAL_SAMPLES = ["G6"]
TSS_BED = os.path.join(DATA_DIR, "K562.positive.bed.gz")

train_pl = [os.path.join(DATA_DIR, f"{s}.pl.bw") for s in TRAIN_SAMPLES]
train_mn = [os.path.join(DATA_DIR, f"{s}.mn.bw") for s in TRAIN_SAMPLES]
train_tss = [TSS_BED] * len(TRAIN_SAMPLES)

val_pl = [os.path.join(DATA_DIR, f"{s}.pl.bw") for s in VAL_SAMPLES]
val_mn = [os.path.join(DATA_DIR, f"{s}.mn.bw") for s in VAL_SAMPLES]
val_tss = [TSS_BED] * len(VAL_SAMPLES)

tss_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=normalization,
    rc_prob=0.5,
    max_jitter=args.window_size // 8,
    tss_centered=True,
    window_size=args.window_size,
    output_stride=args.output_stride,
)
tiled_dataset = NascentDataset(
    train_pl,
    train_mn,
    tss_beds=train_tss,
    transform=normalization,
    rc_prob=0.5,
    max_jitter=args.window_size // 16,
    stride=args.window_size // 2,
    window_size=args.window_size,
    output_stride=args.output_stride,
)
train_loader = MixedBatchLoader(
    tss_dataset,
    tiled_dataset,
    batch_size=args.batch_size,
    tss_fraction=7 / 8,
    num_workers=4,
    pin_memory=True,
)

val_dataset = NascentDataset(
    val_pl,
    val_mn,
    tss_beds=val_tss,
    transform=normalization,
    window_size=args.window_size,
    output_stride=args.output_stride,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))

dilation_tag = "none" if not args.bottleneck_dilations else "-".join(
    str(d) for d in args.bottleneck_dilations
)
strand_tag = "strandfeat" if args.use_strand_features else "rawstrands"
run_name = (
    f"TROGDORResidual_BCE_pw{args.pos_weight:g}_lr{args.lr:g}_"
    f"{args.activation}_{strand_tag}_dil{dilation_tag}_os{args.output_stride}"
)

model = TROGDORResidual(
    name=os.path.join(MODEL_DIR, run_name),
    base_channels=args.base_channels,
    output_stride=args.output_stride,
    context_depth=args.context_depth,
    max_channels=args.max_channels,
    activation=args.activation,
    use_strand_features=args.use_strand_features,
    bottleneck_dilations=args.bottleneck_dilations,
    loss_fn=loss_fn,
).cuda()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)

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
        group="residual_bce",
        config={
            "architecture": "TROGDORResidual",
            "activation": args.activation,
            "bottleneck_dilations": args.bottleneck_dilations,
            "use_strand_features": args.use_strand_features,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "warmup_steps": args.warmup_steps,
            "total_steps": total_steps,
            "loss_fn": "BCEWithLogitsLoss",
            "pos_weight": args.pos_weight,
            "window_size": args.window_size,
            "output_stride": args.output_stride,
            "val_interval": args.val_interval,
        },
    )

model.fit(
    train_loader,
    optimizer,
    val_loader,
    max_epochs=args.max_epochs,
    batch_size=args.batch_size,
    early_stopping=None,
    verbose=True,
    wandb_run=run,
    scheduler=scheduler,
    val_interval=args.val_interval,
)

if run is not None:
    run.finish()
