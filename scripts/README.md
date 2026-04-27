# Scripts

These scripts are for training and benchmarking TROGDOR from scratch.
Most users do not need them — the pre-trained model is bundled with the package.

## Directory layout

```
scripts/
  data/       download_training_data.sh      — fetch K562 PRO/GRO-seq bigWigs from GEO
  train/      train.py                       — train TROGDOR on K562 data
              lr_search.py                   — grid search over learning rates (1e-6 to 1e-3)
              train_residual_bce.py          — train experimental TROGDORResidual
              run_residual_ablations.sh      — run residual architecture ablations
  benchmark/  benchmark.py                   — genome-wide AUROC/AUPRC from a trained model
              benchmark_bw.py                — genome-wide AUROC/AUPRC from a pre-computed prob bigWig
              benchmark_tile_position.py     — compare auPRC for tile-centre vs tile-edge bins
```

## 1. Download training data

Training uses K562 PRO-seq/GRO-seq replicates (G1–G3, G5) from GEO and a
held-out validation replicate (G6). Files are written to `data/`.

```bash
bash scripts/data/download_training_data.sh
```

| Sample | Assay   | GEO accession |
| ------ | ------- | ------------- |
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
validation loss. Early stopping is applied after 5 epochs without improvement.

The LR schedule is linear warmup (500 steps, 1e-8 → 1e-3) followed by
cosine annealing to near zero over the remaining steps.

### Loss function

The default loss is `BCEWithLogitsLoss` (PyTorch built-in). Available built-in alternatives from `chiaroscuro.losses`:

| Function             | Description                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `focal_tversky_loss` | Focal Tversky loss (Abraham & Khan 2019); raises Tversky index to power `1/γ`, emphasising hard missed regions; `α`/`β` control FP/FN weighting. |
| `tversky_loss`       | Plain Tversky index loss without focal re-weighting.                                                                                             |
| `focal_loss`         | Alpha-balanced focal loss (Lin et al. 2017); down-weights easy negatives via `(1−p)^γ`.                                                          |

To override, pass a callable to `loss_fn=` in the `TROGDOR(...)` constructor. For example, we can construct a combined focal + tversky loss function via:

```python
from chiaroscuro.losses import focal_loss, tversky_loss
from chiaroscuro.trogdor import TROGDOR

model = TROGDOR(
    loss_fn=lambda logits, y: focal_loss(logits, y) + 0.5 * tversky_loss(logits, y)
)
```

To tune `α`/`β`/`γ` without writing a wrapper, use `loss_kwargs=`; the training
loop threads these into the loss fn via `functools.partial`:

```python
model = TROGDOR(loss_kwargs={"alpha": 0.3, "beta": 0.7, "gamma": 2.0})
```

### LR search

To find the best peak learning rate before a full training run:

```bash
python scripts/train/lr_search.py
```

Sweeps 7 log-spaced LRs from 1e-6 to 1e-3, training each for 1000 steps with
a flat LR and evaluating val BCE and AUPRC. Results are logged to wandb under
the `lr_search` group.

### Residual architecture experiments

`TROGDORResidual` keeps the same input/output contract as the original model,
but swaps the plain U-Net blocks for residual blocks, learned stride-2
downsampling, optional derived strand channels, and an optional dilated
bottleneck.

Train the default residual candidate:

```bash
conda run -n torch python scripts/train/train_residual_bce.py \
  --pos_weight 500 \
  --lr 1e-3 \
  --activation silu \
  --bottleneck_dilations 1,2,4,8
```

Recommended first ablations:

```bash
conda run -n torch bash scripts/train/run_residual_ablations.sh
```

This runs:

| Variant | Purpose |
| ------- | ------- |
| residual only, no strand features, no dilations | Measures the value of residual blocks + learned downsampling alone |
| residual + strand features, no dilations | Tests plus+minus and plus-minus derived channels |
| residual + strand features + dilated bottleneck | Full default candidate |
| full candidate with `gelu` | Checks whether a smoother transformer-style activation helps |
| full candidate with `relu` | Checks whether the original activation remains best |

Use the same G6 genome-wide benchmark and FDR calibration as the original model
before promoting a residual checkpoint. AUPRC is the first-pass ranking metric;
FDR curves and false positives in broad transcribed regions are the practical
tie-breakers.

Activation choice:

| Activation | Why use it | Main tradeoff |
| ---------- | ---------- | ------------- |
| `silu` | Smooth, non-monotonic-ish gating; often works well in modern conv nets and keeps small negative responses instead of hard-clipping them | Slightly more compute than ReLU |
| `gelu` | Smooth probabilistic gating; can behave similarly to SiLU and is worth one ablation | Slightly more compute; not clearly better for conv signal models |
| `relu` | Fast, simple, and closest to the original TROGDOR baseline | Hard zero for all negatives can discard weak evidence and creates less smooth optimization |

`silu` is the default for `TROGDORResidual` because it is a conservative modern
upgrade for residual convolutional networks: smoother than ReLU, inexpensive,
and less likely than GELU to be a purely stylistic transplant from transformer
defaults.

### Weights & Biases

The script initializes a wandb run automatically. Install wandb first:

```bash
pip install -e ".[dev]"
wandb login
```

Metrics logged per batch: `train/loss`, `train/lr`.
Metrics logged per epoch: `val/loss`, `val/auprc`, `val/dice`.

## 3. Benchmark

Score the whole genome against ground-truth peaks and report AUROC/AUPRC:

```bash
python scripts/benchmark/benchmark.py \
  -M TROGDOR.torch \
  -p data/G6.pl.bw \
  -m data/G6.mn.bw \
  -t data/K562.positive.bed.gz \
  --chroms chr1 chr2 \
  -v
```

| Flag                 | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `-M/--model`         | Path to `.torch` state dict                                            |
| `-p/--pl_bigwig`     | Plus-strand bigWig                                                     |
| `-m/--mn_bigwig`     | Minus-strand bigWig                                                    |
| `-t/--peaks`         | Ground-truth peak BED (gzipped OK)                                     |
| `-d/--device`        | Device (default: `cuda`)                                               |
| `--output_stride`    | Bin size in bp (default: `16`)                                         |
| `--chroms`           | Chromosome whitelist (default: all)                                    |
| `-o/--output_prefix` | Prefix for PDF plots; writes `<prefix>.roc.pdf` and `<prefix>.prc.pdf` |
| `-v/--verbose`       | Show per-chunk tqdm progress bar for each chromosome                   |

In addition to AUROC/AUPRC, the script prints score thresholds corresponding to a range of FPRs and FPRs/TPRs at a range of thresholds.

Expected output on K562 data: AUROC > 0.9, AUPRC meaningfully above the positive rate (~1%).

To benchmark a model that outputs a pre-computed probability bigWig (e.g. a baseline method):

```bash
python scripts/benchmark/benchmark_bw.py \
  -b predictions.bw \
  -t data/K562.positive.bed.gz \
  --chroms chr1 chr2 \
  -v
```

| Flag              | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| `-b/--bigwig`     | Pre-computed probability bigWig                                         |
| `-t/--peaks`      | Ground-truth peak BED (gzipped OK)                                      |
| `--output_stride` | Bin size in bp; probs are max-pooled to this resolution (default: `16`) |
| `--chroms`        | Chromosome whitelist (default: all)                                     |
| `-v/--verbose`    | Print per-chromosome progress                                           |

### Tile position benchmark

Quantifies the AUPRC degradation for bins that fall near chunk boundaries
(edge bins) compared with the same bins predicted from the centre of an
adjacent chunk. Useful for tuning the `--overlap` parameter.

```bash
python scripts/benchmark/benchmark_tile_position.py \
  -M TROGDOR.torch \
  -p data/G6.pl.bw \
  -m data/G6.mn.bw \
  -t data/K562.positive.bed.gz \
  --chroms chr1 chr2 \
  -v
```

| Flag              | Description                                |
| ----------------- | ------------------------------------------ |
| `-M/--model`      | Path to `.torch` state dict                |
| `-p/--pl_bigwig`  | Plus-strand bigWig                         |
| `-m/--mn_bigwig`  | Minus-strand bigWig                        |
| `-t/--peaks`      | Ground-truth peak BED (gzipped OK)         |
| `-d/--device`     | Device (default: `cuda`)                   |
| `--chunk_size`    | Input chunk size in bp (default: `262144`) |
| `--overlap`       | Edge overlap in bp (default: `32768`)      |
| `--output_stride` | Bin size in bp (default: `16`)             |
| `--chroms`        | Chromosome whitelist (default: all)        |
| `-v/--verbose`    | Print per-chromosome progress              |

Output prints the number of comparable bins, the centre auPRC, and the edge
auPRC. A small gap between the two indicates that boundary artefacts are
negligible at the chosen `--overlap`.

### Mixed precision

Training uses bfloat16 autocast by default (`bf16=True`), which requires an
Ampere+ GPU (A100, H100). To disable:

```python
model.fit(..., bf16=False)
```
