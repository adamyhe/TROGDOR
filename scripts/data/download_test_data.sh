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

# K562 mNET-seq
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3518nnn/GSM3518117/suppl/GSM3518117%5FN%5FtCTD%5F1%5FCGGAAT%5F1%5FK562%5Fwildtype.plus.bw -O "$DATA_DIR"/K562_mnetseq_1.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3518nnn/GSM3518117/suppl/GSM3518117%5FN%5FtCTD%5F1%5FCGGAAT%5F1%5FK562%5Fwildtype.minus.bw -O "$DATA_DIR"/K562_mnetseq_1.mn.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3518nnn/GSM3518118/suppl/GSM3518118%5FN%5FtCTD%5F2%5FCTAGCT%5F2%5FK562%5Fwildtype.plus.bw -O "$DATA_DIR"/K562_mnetseq_2.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3518nnn/GSM3518118/suppl/GSM3518118%5FN%5FtCTD%5F2%5FCTAGCT%5F2%5FK562%5Fwildtype.minus.bw -O "$DATA_DIR"/K562_mnetseq_2.mn.bw

bigWigMerge K562_mnetseq_1.pl.bw K562_mnetseq_2.pl.bw K562_mnetseq.pl.bg
bigWigMerge --threshold=-10000000 K562_mnetseq_1.mn.bw K562_mnetseq_2.mn.bw K562_mnetseq.mn.bg
bedGraphToBigWig K562_mnetseq.sort.pl.bg /home2/ayh8/data/hg38.chrom.sizes K562_mnetseq.pl.bw
bedGraphToBigWig K562_mnetseq.sort.mn.bg /home2/ayh8/data/hg38.chrom.sizes K562_mnetseq.mn.bw



# ORIG DATA
# /fs/cbsubscb17/storage/archive/zw355/proj/prj10-dreg/GM12878
# /fs/cbsubscb17/storage/archive/zw355/proj/prj10-dreg/K562

