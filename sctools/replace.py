# -*- coding: utf-8 -*-
#! /usr/bin/env python

import re
from Bio import SeqIO
import gzip


def snpCorrect(genome, chrom, pos, ref, alt):
    """
    replace reference base with alternate base, return list
    genome is a dictionary with key == chromosome name, value == list of characters
    Assumes pos numbering starts at 1 (python starts at 0)
    """
    ref = ref.upper()
    alt = alt.upper()
    c = {"M": "Mt", "C": "Pt"}  # chromosomes named differently in tair10 refrence and SNP set
    if ref == alt:
        raise Exception("Error: SNP base is same as reference base")
    pos = int(pos)
    if chrom in c.keys():
        chrom = c[chrom]
    if genome[chrom][pos-1].upper() == ref:
        genome[chrom][pos-1] = alt
    else:
        print('Warning: reference genome sequence does not match at position {}:{}'.format(str(chrom), str(pos)))


def produceString(key, dic):
    string = ">" + key + "\n"
    seq = "".join(dic[key])
    seq = re.sub("(.{60})", "\\1\n", seq, 0, re.DOTALL)  # 60 bases per line
    return string + seq + "\n"

def run_replace(genome, snp, outfile):
    """
    Replace reference genome bases with ambiguous base codes at
    SNP positions

    Substitutes the appropriate IUPAC base code that encodes both the reference genome
    base and alternative allele base at the correct postitions in the genome. Will
    give a warning if the reference sequence does not match the expected base.

    Parameters
    ----------
    genome : str
        Name of reference genome fasta files
    snp : str
        Name of SNP file
    outfile : str
        Name of output fasta file
    """
    chroms = {}
    for record in SeqIO.parse(genome, "fasta"):
        chroms[record.id] = list(record.seq)

    iupac = {
        'ag': 'r',
        'ga': 'r',
        'ct': 'y',
        'tc': 'y',
        'gc': 's',
        'cg': 's',
        'at': 'w',
        'ta': 'w',
        'gt': 'k',
        'tg': 'k',
        'ac': 'm',
        'ca': 'm'
        }

    if snp.endswith("gz"):
        infile = gzip.open(snp, "rb")
    else:
        infile = open(snp, "r")
    for line in infile:
        line  = line.rsplit()
        chrom = line[0].strip("chr")
        pos = int(line[1])
        ref = line[2]
        alt = line[3]
        comb = (ref+alt).lower()
        try:
            iupac[comb]
        except KeyError:
            pass
        else:
            ambig = iupac[comb]
            snpCorrect(chroms, chrom, pos, ref, ambig)
    infile.close()

    with open(outfile, "w") as outfile:
        # to get chromsomes in proper order, need to write one-by-one
        outfile.write(produceString("1", chroms))
        outfile.write(produceString("2", chroms))
        outfile.write(produceString("3", chroms))
        outfile.write(produceString("4", chroms))
        outfile.write(produceString("5", chroms))
        outfile.write(produceString("Mt", chroms))
        outfile.write(produceString("Pt", chroms))
