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

# K562 GRO-seq (same source as training sample G2, named for Figure 1B/C benchmarks)
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480325/suppl/GSM1480325%5FK562%5FGROseq%5Fplus.bigWig -O "$DATA_DIR"/K562_groseq.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1480nnn/GSM1480325/suppl/GSM1480325%5FK562%5FGROseq%5Fminus.bigWig -O "$DATA_DIR"/K562_groseq.mn.bw

# Jurkat PRO-seq/ChRO-seq
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309955/suppl/GSM3309955%5F5587%5F5598%5F24204%5FHGC2FBGXX%5FJ%5FNUC%5FTTAGGC%5FR1%5Fplus%2Ebw -O "$DATA_DIR"/Jurkat_PROseq.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309955/suppl/GSM3309955%5F5587%5F5598%5F24204%5FHGC2FBGXX%5FJ%5FNUC%5FTTAGGC%5FR1%5Fminus%2Ebw -O "$DATA_DIR"/Jurkat_PROseq.mn.bw

wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309958/suppl/GSM3309958%5FJurkat%5FChRO%5FRNase%5Fplus%2Ebw -O "$DATA_DIR"/Jurkat_leChROseq.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309958/suppl/GSM3309958%5FJurkat%5FChRO%5FRNase%5Fminus%2Ebw -O "$DATA_DIR"/Jurkat_leChROseq.mn.bw

wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309957/suppl/GSM3309957%5FJurkat%5FChRO%5FNoRNase%5Fplus%2Ebw -O "$DATA_DIR"/Jurkat_ChROseq_1.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309957/suppl/GSM3309957%5FJurkat%5FChRO%5FNoRNase%5Fminus%2Ebw -O "$DATA_DIR"/Jurkat_ChROseq_1.mn.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309956/suppl/GSM3309956%5F5587%5F5598%5F24205%5FHGC2FBGXX%5FJ%5FCHR%5FTGACCA%5FR1%5Fplus%2Ebw -O "$DATA_DIR"/Jurkat_ChROseq_2.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3309nnn/GSM3309956/suppl/GSM3309956%5F5587%5F5598%5F24205%5FHGC2FBGXX%5FJ%5FCHR%5FTGACCA%5FR1%5Fminus%2Ebw -O "$DATA_DIR"/Jurkat_ChROseq_2.mn.bw

HG19_SIZES="${DATA_DIR}/hg19.chrom.sizes"

bigWigMerge "$DATA_DIR"/Jurkat_ChROseq_1.pl.bw "$DATA_DIR"/Jurkat_ChROseq_2.pl.bw "$DATA_DIR"/Jurkat_ChROseq.pl.bg
bigWigMerge -threshold=-10000000 "$DATA_DIR"/Jurkat_ChROseq_1.mn.bw "$DATA_DIR"/Jurkat_ChROseq_2.mn.bw "$DATA_DIR"/Jurkat_ChROseq.mn.bg
sort -k1,1 -k2,2n "$DATA_DIR"/Jurkat_ChROseq.pl.bg > "$DATA_DIR"/Jurkat_ChROseq.sort.pl.bg
sort -k1,1 -k2,2n "$DATA_DIR"/Jurkat_ChROseq.mn.bg > "$DATA_DIR"/Jurkat_ChROseq.sort.mn.bg
bedGraphToBigWig "$DATA_DIR"/Jurkat_ChROseq.sort.pl.bg "$HG19_SIZES" "$DATA_DIR"/Jurkat_ChROseq.pl.bw
bedGraphToBigWig "$DATA_DIR"/Jurkat_ChROseq.sort.mn.bg "$HG19_SIZES" "$DATA_DIR"/Jurkat_ChROseq.mn.bw

# K562 mNET-seq
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3518nnn/GSM3518117/suppl/GSM3518117%5FN%5FtCTD%5F1%5FCGGAAT%5F1%5FK562%5Fwildtype.plus.bw -O "$DATA_DIR"/K562_mnetseq_1.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3518nnn/GSM3518117/suppl/GSM3518117%5FN%5FtCTD%5F1%5FCGGAAT%5F1%5FK562%5Fwildtype.minus.bw -O "$DATA_DIR"/K562_mnetseq_1.mn.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3518nnn/GSM3518118/suppl/GSM3518118%5FN%5FtCTD%5F2%5FCTAGCT%5F2%5FK562%5Fwildtype.plus.bw -O "$DATA_DIR"/K562_mnetseq_2.pl.bw
wget --no-check-certificate https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3518nnn/GSM3518118/suppl/GSM3518118%5FN%5FtCTD%5F2%5FCTAGCT%5F2%5FK562%5Fwildtype.minus.bw -O "$DATA_DIR"/K562_mnetseq_2.mn.bw

HG38_SIZES="${DATA_DIR}/hg38.chrom.sizes"

bigWigMerge "$DATA_DIR"/K562_mnetseq_1.pl.bw "$DATA_DIR"/K562_mnetseq_2.pl.bw "$DATA_DIR"/K562_mnetseq.pl.bg
bigWigMerge -threshold=-10000000 "$DATA_DIR"/K562_mnetseq_1.mn.bw "$DATA_DIR"/K562_mnetseq_2.mn.bw "$DATA_DIR"/K562_mnetseq.mn.bg
sort -k1,1 -k2,2n "$DATA_DIR"/K562_mnetseq.pl.bg > "$DATA_DIR"/K562_mnetseq.sort.pl.bg
sort -k1,1 -k2,2n "$DATA_DIR"/K562_mnetseq.mn.bg > "$DATA_DIR"/K562_mnetseq.sort.mn.bg
bedGraphToBigWig "$DATA_DIR"/K562_mnetseq.sort.pl.bg "$HG38_SIZES" "$DATA_DIR"/K562_mnetseq.pl.bw
bedGraphToBigWig "$DATA_DIR"/K562_mnetseq.sort.mn.bg "$HG38_SIZES" "$DATA_DIR"/K562_mnetseq.mn.bw

# HeLa mNET-seq goes here.

