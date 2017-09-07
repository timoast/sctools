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
derived from each genotype in each cell using the `sctools
countsnps` command.

```
$ sctools countsnps -h
usage: sctools countsnps [-h] -b BAM -s SNP -o OUTPUT [-c CELLS] [-p NPROC]

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

### Genotype cells

If your data is a mix of two genotypes, you can assign genotype labels to cells based
SNP counts calculated using `sctools countsnps`, using the `sctools genotype` command.

This runs an iterative density-based clustering of cells based on SNP UMIs using DBSCAN.

```
$ sctools genotype -h
usage: sctools genotype [-h] -s INFILE -o OUTFILE [-n NPROC] [-p] [-d]
                        [--eps_background EPS_BACKGROUND]
                        [--eps_cells EPS_CELLS]
                        [--min_samples_background MIN_SAMPLES_BACKGROUND]
                        [--min_samples_cells MIN_SAMPLES_CELLS]

Genotype cells based on SNP UMI counts.

optional arguments:
  -h, --help            show this help message and exit
  -s INFILE, --infile INFILE
                        SNP UMI counts for each genotype
  -o OUTFILE, --outfile OUTFILE
                        Name for output text file
  -n NPROC, --nproc NPROC
                        Number of processors (default = 1)
  -p, --plot            Plot results
  -d, --downsample      Do not downsample cells before detecting background
                        cluster
  --eps_background EPS_BACKGROUND
                        DBSCAN epsilon value for background cell clustering
  --eps_cells EPS_CELLS
                        DBSCAN epsilon value for background cell clustering
  --min_samples_background MIN_SAMPLES_BACKGROUND
                        Minimum number of cells in each cluster for background
                        cell clustering
  --min_samples_cells MIN_SAMPLES_CELLS
                        Minimum number of cells in each cluster for background
                        cell clustering
```

### Filter barcodes

If you want to extract reads from a subset of cell barcodes, you can
do so using the `sctools filterbarcodes` command.

```
$ sctools filterbarcodes -h
usage: sctools filterbarcodes [-h] -b BAM -c CELLS -o OUTPUT [-s]
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

### Count edited transcripts

Edited RNA transcripts can be counted per editing position per cell using the `sctools countedited` command.

```
$ sctools countedited -h    [master]
usage: sctools countedited [-h] -b BAM -e EDIT -o OUTPUT [-c CELLS]
                              [-p NPROC]

Count edited transcripts per gene per cell in single-cell RNA data. Output is
a matrix of positions by cells.

optional arguments:
  -h, --help            show this help message and exit
  -b BAM, --bam BAM     Input bam file (must be indexed)
  -e EDIT, --edit EDIT  File with edited base coordinates. Needs chromosome,
                        position, reference, alternate as first four columns
  -o OUTPUT, --output OUTPUT
                        Name for output text file
  -c CELLS, --cells CELLS
                        File containing cell barcodes to count edited bases
                        for. Can be gzip compressed (optional)
  -p NPROC, --nproc NPROC
                        Number of processors (default = 1)
```

## Import sctools as a python module

You can also use the sctools functions interactively or in python scripts by importing sctools as a module.

For example, to determine cell genotypes more interactively (rather than running `sctools genotype` on the command line), you can do something like this:

```python
from sctools import genotype
import pandas as pd

data = pd.read_csv("snps.tsv")

geno = genotype.Genotype(data)                 # initiate the genotype class
geno.transform_snps()                          # log transform SNP UMI counts
geno.filter_low_count(min_log10_count=1)       # remove cells with less than log10(UMI) = 1
geno.detect_background(eps=1, min_samples=500) # detect background cell cluster
geno.segment_cells()                           # downsample cells to equal numbers for each genotype
geno.find_clusters(eps=0.3, min_samples=100)   # detect cell clusters
geno.label_barcodes()                          # attach genotype labels to cell barcodes
```
