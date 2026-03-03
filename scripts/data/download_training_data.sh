#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../data"
mkdir -p "$DATA_DIR"

# G1
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480327/suppl/GSM1480327%5FK562%5FPROseq%5Fplus.bw -O "$DATA_DIR"/G1.pl.bw
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480327/suppl/GSM1480327%5FK562%5FPROseq%5Fminus.bw -O "$DATA_DIR"/G1.mn.bw

# G2
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480325/suppl/GSM1480325%5FK562%5FGROseq%5Fplus.bigWig -O "$DATA_DIR"/G2.pl.bw
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480325/suppl/GSM1480325%5FK562%5FGROseq%5Fminus.bigWig -O "$DATA_DIR"/G2.mn.bw

# G3
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3452nnn/GSM3452725/suppl/GSM3452725%5FK562%5FNuc%5FNoRNase%5Fplus.bw -O "$DATA_DIR"/G3.pl.bw
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3452nnn/GSM3452725/suppl/GSM3452725%5FK562%5FNuc%5FNoRNase%5Fminus.bw -O "$DATA_DIR"/G3.mn.bw

# G5
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE89nnn/GSE89230/suppl/GSE89230%5FNormalized%5FPRO%2Dseq%5FK562%5Fcombined%5Freplicates%5FNHS%5FplusStrand.bigWig -O "$DATA_DIR"/G5.pl.bw
wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE89nnn/GSE89230/suppl/GSE89230%5FNormalized%5FPRO%2Dseq%5FK562%5Fcombined%5Freplicates%5FNHS%5FminusStrand.bigWig -O "$DATA_DIR"/G5.mn.bw

# G6
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2545nnn/GSM2545324/suppl/GSM2545324%5F6045%5F7157%5F27170%5FHNHKJBGXX%5FK562%5F0min%5Fcelastrol10uM%5Frep1%5FGB%5FATCACG%5FR1%5Fplus.primary.bw -O "$DATA_DIR"/G6.pl.bw
wget https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2545nnn/GSM2545324/suppl/GSM2545324%5F6045%5F7157%5F27170%5FHNHKJBGXX%5FK562%5F0min%5Fcelastrol10uM%5Frep1%5FGB%5FATCACG%5FR1%5Fminus.primary.bw -O "$DATA_DIR"/G6.mn.bw

# Positives peaks
rsync --progress ftp://cbsuftp.tc.cornell.edu/danko/hub/TROGDOR/K562.positive.bed.gz "$DATA_DIR"/K562.positive.bed.gz