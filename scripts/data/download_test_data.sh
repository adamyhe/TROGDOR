#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../data"
mkdir -p "$DATA_DIR"

# G7
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2545nnn/GSM2545325/suppl/GSM2545325%5F6045%5F7157%5F27176%5FHNHKJBGXX%5FK562%5F0min%5Fcelastrol10uM%5Frep2%5FGB%5FCAGATC%5FR1%5Fplus.primary.bw -O "$DATA_DIR"/G7.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2545nnn/GSM2545325/suppl/GSM2545325%5F6045%5F7157%5F27176%5FHNHKJBGXX%5FK562%5F0min%5Fcelastrol10uM%5Frep2%5FGB%5FCAGATC%5FR1%5Fminus.primary.bw -O "$DATA_DIR"/G7.mn.bw

# GM12878 GRO-seq
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480326/suppl/GSM1480326%5FGM12878%5FGROseq%5Fplus.bigWig -O "$DATA_DIR"/GM12878_groseq.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480326/suppl/GSM1480326%5FGM12878%5FGROseq%5Fminus.bigWig -O "$DATA_DIR"/GM12878_groseq.mn.bw

wget ftp://cbsuftp.tc.cornell.edu/danko/hub/TROGDOR/GM12878.positive.bed.gz -P "$DATA_DIR"

# ORIG DATA
# /fs/cbsubscb17/storage/archive/zw355/proj/prj10-dreg/GM12878
# /fs/cbsubscb17/storage/archive/zw355/proj/prj10-dreg/K562

