#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../data"
mkdir -p "$DATA_DIR"

# GM12878 groHMM AND DNase peaks
wget ftp://cbsuftp.tc.cornell.edu/danko/hub/TROGDOR/GM12878.positive.bed.gz -P "$DATA_DIR"

# K562 groHMM AND DNase peaks
wget ftp://cbsuftp.tc.cornell.edu/danko/hub/TROGDOR/K562.positive.bed.gz -P "$DATA_DIR"

# Download ENCODE cCREs

# Filter for PLS/ELS

# Convert go hg19