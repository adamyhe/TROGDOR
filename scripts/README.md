# Scripts

These scripts are for training and benchmarking TROGDOR from scratch.
Most users do not need them — the pre-trained model is bundled with the package.

## Directory layout

```
scripts/
  data/     download_training_data.sh  — fetch K562 PRO/GRO-seq bigWigs from GEO
  train/    train.py                   — train TROGDOR on K562 data
            lr_search.py               — grid search over learning rates (1e-6 to 1e-3)
  benchmark/                           — (planned) benchmarking utilities
```

## 1. Download training data

Training uses K562 PRO-seq/GRO-seq replicates (G1–G3, G5) from GEO and a
held-out validation replicate (G6). Files are written to `data/`.

```bash
bash scripts/data/download_training_data.sh
```

| Sample | Assay   | GEO accession |
|--------|---------|---------------|
| G1     | PRO-seq | GSM1480327    |
| G2     | GRO-seq | GSM1480325    |
| G3     | GRO-seq | GSM3452725    |
| G5     | PRO-seq | GSE89230      |
| G6     | PRO-seq | GSM2545324    |

Positive TIR peaks are fetched from the Danko lab FTP (`K562.positive.bed.gz`).

## 2. Train

```bash
python scripts/train/train.py
```

Trains on G1/G2/G3/G5 with a 7:1 ratio of TSS-centered to genome-tiled
windows per batch, validates on G6, and saves the best checkpoint by
validation BCE. Early stopping is applied after 5 epochs without improvement.

The LR schedule is linear warmup (500 steps, 1e-8 → 1e-3) followed by
cosine annealing to near zero over the remaining steps.

### LR search

To find the best peak learning rate before a full training run:

```bash
python scripts/train/lr_search.py
```

Sweeps 7 log-spaced LRs from 1e-6 to 1e-3, training each for 1000 steps with
a flat LR and evaluating val BCE and AUPRC. Results are logged to wandb under
the `lr_search` group.

### Weights & Biases

The script initializes a wandb run automatically. Install wandb first:

```bash
pip install -e ".[dev]"
wandb login
```

Metrics logged per batch: `train/bce`, `train/lr`.
Metrics logged per epoch: `val/bce`, `val/auprc`, `val/dice`.

### Mixed precision

Training uses bfloat16 autocast by default (`bf16=True`), which requires an
Ampere+ GPU (A100, H100). To disable:

```python
model.fit(..., bf16=False)
```
