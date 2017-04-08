# sctools

Tools for single-cell RNA-seq analysis

## Installation

```bash
git clone https://github.com/timoast/sctools.git
cd sctools
pip install -r requirements.txt
python setup.py install
```

## Features

### Count SNPs

If your data contains a mixture of two genotypes, you can count UMIs
derived from each genotype in each cell using the `sctools.py
countsnps` command.

```
$ sctools.py countsnps -h
usage: sctools.py countsnps [-h] -b BAM -s SNP -o OUTPUT [-c CELLS] [-p NPROC]

Count reference and alternate SNPs per cell in single-cell RNA data

optional arguments:
  -h, --help            show this help message and exit
  -b BAM, --bam BAM     Input bam file (must be indexed)
  -s SNP, --snp SNP     File with SNPs. Needs chromosome, position, reference,
                        alternate as first four columns
  -o OUTPUT, --output OUTPUT
                        Name for output text file
  -c CELLS, --cells CELLS
                        File containing cell barcodes
  -p NPROC, --nproc NPROC
                        Number of processors (default = 1)
```

### Filter barcodes

If you want to extract reads from a subset of cell barcodes, you can
do so using the `sctools.py filterbarcodes` command.

```
$ sctools.py filterbarcodes -h
usage: sctools.py filterbarcodes [-h] -b BAM -c CELLS -o OUTPUT [-s]
                                 [-p NPROC]

Filter reads based on input list of cell barcodes

optional arguments:
  -h, --help            show this help message and exit
  -b BAM, --bam BAM     Input bam file (must be indexed)
  -c CELLS, --cells CELLS
                        File containing cell barcodes. Can be gzip compressed
  -o OUTPUT, --output OUTPUT
                        Name for output text file
  -s, --sam             Output sam format (default bam output)
  -p NPROC, --nproc NPROC
                        Number of processors (default = 1)
```

