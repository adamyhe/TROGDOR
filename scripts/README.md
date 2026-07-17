# Scripts

These scripts are for training and benchmarking TROGDOR from scratch.
Most users do not need them — the pre-trained model is bundled with the package.

## Directory layout

```
scripts/
  data/       download_training_data.sh      — fetch K562 PRO/GRO-seq bigWigs from GEO
  train/      train.py                       — train TROGDOR on K562 data (--loss, --tss_bed, etc.)
              lr_search.py                   — grid search over learning rates (1e-6 to 1e-3)
  benchmark/  benchmark.py                   — genome-wide AUROC/AUPRC from a trained model
              benchmark_bw.py                — genome-wide AUROC/AUPRC from a pre-computed prob bigWig
              benchmark_tile_position.py     — compare auPRC for tile-centre vs tile-edge bins
              compare_peaks.py               — peak-level precision/recall/F1/Jaccard against a truth BED
              truth_panel.py                 — multi-reference label ambiguity report
              frip.py                        — raw/normalized FRIP from stranded bigWigs and a peak BED
              logit_dist.py                  — logit score distribution diagnostic
              infp_filter.py                 — dREG-style informative-positions mask (see note below)
              fdr_dreg.py                    — empirical FDR estimation for dREG scored BEDs
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
validation AUPRC. Early stopping is available (`--early_stopping N`) but
disabled by default, matching the current production recipe.

The LR schedule is linear warmup (500 steps, 1e-8 → target LR) followed by
cosine annealing to near zero over the remaining steps.

**Why validate on G6 instead of holding out chromosomes**: G1–G6 are all K562,
but from different labs, different assays (GRO-seq vs. PRO-seq), and
different read depths — the split tests generalization across
labs/protocols/depth, which is the actual deployment scenario (at inference
time TROGDOR is run on other cell types, other assays like ChRO-seq/mNET-seq,
and evaluated against other peak/functional-mark sets entirely). A
chromosome-based holdout would instead test generalization to unseen loci
within the same handful of same-cell-type experiments, which is not the
generalization axis this tool actually needs.

| Flag              | Default                     | Description                                                              |
| ----------------- | ---------------------------- | ------------------------------------------------------------------------ |
| `--loss`          | `bce`                        | `bce`, `focal_tversky`, `tversky`, `focal`, or `focal+tversky`           |
| `--pos_weight`    | `500`                         | Positive class weight for `BCEWithLogitsLoss`; only used when `--loss=bce`. `0` disables reweighting |
| `--tss_bed`       | `K562.positive.bed.gz`        | TSS/TIR label BED shared by train and val samples (see note below)      |
| `--lr`            | `1e-3`                        | Learning rate                                                            |
| `--max_epochs`    | `20`                          | Maximum training epochs                                                  |
| `--early_stopping`| off                           | Stop after N epochs without validation-AUPRC improvement                |
| `--warmup_steps`  | `500`                         | Linear warmup steps                                                      |
| `--weight_decay`  | `1e-4`                        | AdamW weight decay                                                       |
| `--window_size`   | `2**18` (262144 bp)           | Training window size in bp                                               |
| `--batch_size`    | `64`                          | Training batch size                                                      |
| `--val_interval`  | off                           | Validate every N steps in addition to epoch end                         |
| `--run_name`      | auto-generated                | Override the checkpoint/wandb run name                                   |
| `--no_wandb`      | off                           | Disable Weights & Biases logging                                        |

### Loss function

`--loss bce` (the default) is the current production recipe: `BCEWithLogitsLoss`
with `pos_weight=500` to compensate for how rare TIR bins are within a
training window (this is not the same as the ~1% genome-wide positive rate —
even TSS-centered windows are >99.9% negative bins locally, since a training
window is far wider than a TIR). The other choices from `chiaroscuro.losses`
were tried and have not outperformed BCE+`pos_weight` in benchmarking so far:

| Function             | Description                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `focal_tversky_loss` | Focal Tversky loss (Abraham & Khan 2019); raises Tversky index to power `1/γ`, emphasising hard missed regions; `α`/`β` control FP/FN weighting. |
| `tversky_loss`       | Plain Tversky index loss without focal re-weighting.                                                                                             |
| `focal_loss`         | Alpha-balanced focal loss (Lin et al. 2017); down-weights easy negatives via `(1−p)^γ`.                                                          |

`--loss focal+tversky` combines the last two with equal weight. To tune
`α`/`β`/`γ`, or to try a loss combination not exposed as a `--loss` choice,
edit `NON_BCE_LOSSES`/the `loss_fn=`/`loss_kwargs=` construction in
`scripts/train/train.py` directly — `TROGDOR`'s constructor threads
`loss_kwargs` into the loss fn via `functools.partial`:

```python
model = TROGDOR(loss_kwargs={"alpha": 0.3, "beta": 0.7, "gamma": 2.0})
```

`--tss_bed` controls which label source defines "positive." The default
(`K562.positive.bed.gz`, dREG-derived) matches the shipped checkpoint. Two
other sources have been tried — PINTS peaks (`ENCSR220XSM_peaks.hg19.bed`)
and ENCODE SCREEN cCREs (`K562_ENCODE_prom_enh.hg19.bed`) — but these are
different *definitions* of a true positive, not just different files: a
model trained against one label source scores much better against that same
source at test time than against the others, so `--pos_weight` and other
defaults tuned for one label source are not guaranteed to transfer to
another.

### LR search

To find the best peak learning rate before a full training run:

```bash
python scripts/train/lr_search.py
```

Sweeps 7 log-spaced LRs from 1e-6 to 1e-3, training each for 1000 steps with
a flat LR and evaluating val BCE and AUPRC. Results are logged to wandb under
the `lr_search` group.

### Weights & Biases

The script initializes a wandb run automatically. Install the development
dependencies as described in [`../CONTRIBUTING.md`](../CONTRIBUTING.md), then
log in:

```bash
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

### Truth panel reports

`truth_panel.py` evaluates the same scored bigWig and call BED against multiple
orthogonal reference sets and stratifies unmatched calls. The manifest is a TSV
with required columns:

```text
cell_type  prob_bw  calls_bed  reference_name  reference_bed
```

Optional columns add ambiguity categories:

```text
tss_bed  promoter_bed  enhancer_bed  bidirectional_bed  gene_body_bed  blacklist_bed  reproducible_bed
```

Run:

```bash
python scripts/benchmark/truth_panel.py \
  --manifest truth_panel.tsv \
  --chrom_sizes hg38.chrom.sizes \
  --output_prefix panel_report \
  --chroms chr1 chr2
```

Outputs:

| File | Description |
| ---- | ----------- |
| `panel_report.reference_metrics.tsv` | Per-cell-type/per-reference bin, peak, and center-window metrics |
| `panel_report.unmatched_categories.tsv` | Unmatched-call categories such as near-TSS, enhancer, gene-body, broad, reproducible, blacklist, and unsupported |

### Mixed precision

Training uses bfloat16 autocast by default (`bf16=True`), which requires an
Ampere+ GPU (A100, H100). To disable:

```python
model.fit(..., bf16=False)
```
