#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../data"
mkdir -p "$DATA_DIR"

# hg38 chromosome sizes
wget -q https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes \
    -O "$DATA_DIR/hg38.chrom.sizes"

# hg38 -> hg19 liftOver chain
wget --timestamping \
    'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz' \
    -O "$DATA_DIR/hg38ToHg19.over.chain.gz"
