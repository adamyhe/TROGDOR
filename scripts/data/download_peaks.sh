#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../data"
mkdir -p "$DATA_DIR"

# GM12878 groHMM AND DNase peaks
wget ftp://cbsuftp.tc.cornell.edu/danko/hub/TROGDOR/GM12878.positive.bed.gz -P "$DATA_DIR"

# K562 groHMM AND DNase peaks
wget ftp://cbsuftp.tc.cornell.edu/danko/hub/TROGDOR/K562.positive.bed.gz -P "$DATA_DIR"

# Download ENCODE cCREs
# GM12878
wget https://downloads.wenglab.org/Registry-V4/ENCFF428XFI_ENCFF280PUF_ENCFF469WVA_ENCFF644EEX.bed -O "$DATA_DIR/K562_ENCODE_cCRE.hg38.bed"
# K562
wget https://downloads.wenglab.org/Registry-V4/ENCFF414OGC_ENCFF806YEZ_ENCFF849TDM_ENCFF736UDR.bed -O "$DATA_DIR/GM12878_ENCODE_cCRE.hg38.bed"
# HeLa
wget https://downloads.wenglab.org/Registry-V4/ENCFF757GHL_ENCFF432PYK_ENCFF658XKZ_ENCFF179RSE.bed -O "$DATA_DIR/HeLa_ENCODE_cCRE.hg38.bed"

# Filter for PLS/ELS
grep -E "PLS|ELS" "$DATA_DIR/GM12878_ENCODE_cCRE.hg38.bed" > "$DATA_DIR/GM12878_ENCODE_prom_enh.hg38.bed"
grep -E "PLS|ELS" "$DATA_DIR/K562_ENCODE_cCRE.hg38.bed"   > "$DATA_DIR/K562_ENCODE_prom_enh.hg38.bed"
grep -E "PLS|ELS" "$DATA_DIR/HeLa_ENCODE_cCRE.hg38.bed"   > "$DATA_DIR/HeLa_ENCODE_prom_enh.hg38.bed"

# Convert hg38 -> hg19 via liftOver
# Requires liftOver and hg38ToHg19.over.chain.gz (UCSC)
CHAIN="${DATA_DIR}/hg38ToHg19.over.chain.gz"
wget -q https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz -O "$CHAIN"

for CELL in GM12878 K562 HeLa; do
    liftOver \
        "$DATA_DIR/${CELL}_ENCODE_prom_enh.hg38.bed" \
        "$CHAIN" \
        "$DATA_DIR/${CELL}_ENCODE_prom_enh.hg19.bed" \
        "$DATA_DIR/${CELL}_ENCODE_prom_enh.hg19.unmapped.bed"
done