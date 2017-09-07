#! /usr/bin/env python

"""
single cell tools
"""

from __future__ import absolute_import
import pysam
import gzip
from multiprocessing import Pool
import functools
import random
import string
from glob import glob
import os
import time
from subprocess import call
import pandas as pd
import numpy as np
from sctools import genotype


def log_info(func):
    @functools.wraps(func)
    def wrapper(args):
        print("Function {} called with the following arguments:\n".format(func.__name__))
        for arg in vars(args):
            print(str(arg) + '\t' + str(getattr(args, arg)))
        t1 = time.time()
        func(args)
        t2 = time.time()
        elapsed = [round(x, 2) for x in divmod(t2-t1, 60)]
        print("\nFunction completed in  {} m {} s\n".format(elapsed[0], elapsed[1]))
    return wrapper


def chunk_bam(bamfile, nproc):
    """
    chunk file into n chunks for multicore
    """
    chrom_lengths = bamfile.lengths
    chunksize = sum(chrom_lengths) / int(nproc)
    intervals = []
    for x in range(1, nproc+1):
        position = chunksize*x
        intervals.append(find_chromosome_break(position, chrom_lengths, 0))
    return add_start_coords(intervals, chrom_lengths, bamfile)


def add_start_coords(intervals, chrom_lengths, bamfile):
    """
    given the intervals that will be handled by each core,
    break into genomic regions (removing chromosome-spanning intervals)
    """
    # first add the start coordinate
    intervals = [[1,0]] + intervals

    # now get start stop postions for all intervals
    ranges = [intervals[x-1] + intervals[x] for x in range(1, len(intervals))]

    # populate dictonary of genomic intervals
    # each key is a list of ranges
    # one key = one process
    d = {}
    x = 0
    for i in ranges:
        x += 1
        if i[0] == i[2]:  # same chromosome
            d[x] = [(bamfile.get_reference_name(i[0]-1), i[1], i[3])]
        else: # range spans one or more chromosomes
            d[x] = [(bamfile.get_reference_name(i[0]-1), i[1], chrom_lengths[i[0] - 1])] # record the first interval
            nchrom = i[2] - i[0] # number of chromosomes spanned by the range
            for y in range(nchrom - 1): # will be nothing unless we need to completely cover a chromosome
                d[x].append((bamfile.get_reference_name(i[0] + y), 0, chrom_lengths[i[0] + y]))
            # now record the last bit
            d[x].append((bamfile.get_reference_name(i[2] -1), 0, i[3]))
    return(d)


def find_chromosome_break(position, chromosomes, current_chrom):
    assert position <= sum(chromosomes), "position past end of genome"
    if position <= chromosomes[current_chrom]:
        return [current_chrom + 1, position]
    else:
        position = position - chromosomes[current_chrom]
        return find_chromosome_break(position, chromosomes, current_chrom + 1)


def iterate_reads(intervals, bam, sam, output, cb):
    inputBam = pysam.AlignmentFile(bam, 'rb')
    ident = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    if sam:
        outputBam = pysam.AlignmentFile(output + ident, "w", template=inputBam)
    else:
        outputBam = pysam.AlignmentFile(output + ident, 'wb', template=inputBam)
    for i in intervals:
        for r in inputBam.fetch(i[0], i[1], i[2]):
            cell_barcode, _ = scan_tags(r.tags)
            if cell_barcode is not None:
                if cell_barcode[:-2] in cb:
                    outputBam.write(r)
    outputBam.close()
    inputBam.close()
    return output + ident


def read_snps(snps):
    """
    Input SNP coordinates and bases
    Return list of chromosome, position, reference base, alternate base
    """
    data = []
    if snps.endswith(".gz"):
        infile = gzip.open(snps, "rb")
    else:
        infile = open(snps, "r")
    for line in infile:
        line = line.rsplit()
        chromosome = line[0].strip("chr")
        position = int(line[1])
        ref = line[2]
        alt = line[3]
        data.append([chromosome, position, ref, alt])
    infile.close()
    return(data)


def scan_tags(tags):
    """
    Input bam tags
    Return UMI and cell barcode sequences
    """
    cell_barcode = None
    umi = None
    for tag in tags:
        if tag[0] == "CB":
            cell_barcode = tag[1]
        elif tag[0] == "UB":
            umi = tag[1]
        else:
            pass
    return cell_barcode, umi


def get_genotype(read, ref_position, snp_position, cigar_tuple):
    """
    Input read position, read sequence, SNP position, cigar
    Return the base in read that is aligned to the SNP position
    """
    cigar = { # alignement code, ref shift, read shift
        0: ["M", 1, 1], # match, progress both
        1: ["I", 0, 1], # insertion, progress read not reference
        2: ["D", 1, 0], # deletion, progress reference not read
        3: ["N", 1, 0], # skipped, progress reference not read
        4: ["S", 0, 1], # soft clipped, progress read not reference
        5: ["H", 0, 0], # hard clipped, progress neither
        6: ["P", 0, 0], # padded, do nothing (not used)
        7: ["=", 1, 1], # match, progress both
        8: ["X", 1, 1]  # mismatch, progress both
    }
    read_position = 0
    for i in cigar_tuple:
        ref_bases = cigar[i[0]][1] * i[1]
        read_bases = cigar[i[0]][2] * i[1]
        if ref_bases == 0 and read_bases == 0:  # this shouldn't ever happen
            pass
        elif ref_bases == 0 and read_bases > 0: # clipped bases or insertion relative to reference
            read_position += read_bases
            if ref_position == snp_position:   # only happens when first aligned base is the SNP
                return read[read_position - 1]
        elif read_bases == 0 and ref_bases > 0:
            ref_position += ref_bases
            if ref_position > snp_position:  # we've gone past the SNP
                return None
        else:
            if ref_position + ref_bases > snp_position: # we pass snp
                shift = snp_position - ref_position
                return read[read_position + shift - 1]
            elif ref_position + ref_bases < snp_position:
                ref_position += ref_bases
                read_position += read_bases
            else:
                return read[read_position + read_bases - 1]
    return None


def add_genotype(cell_genotypes, cell_barcode, umi, genotype):
    """
    Append genotype information for cell and UMI to dictionary
    return modified cell_genotypes dictionary
    """
    try:
        cell_genotypes[cell_barcode]
    except KeyError:
        # haven't seen the cell, must be new UMI
        cell_genotypes[cell_barcode] = {umi: [genotype]}
    else:
        try:
            cell_genotypes[cell_barcode][umi]
        except KeyError:
            cell_genotypes[cell_barcode][umi] = [genotype]
        else:
            cell_genotypes[cell_barcode][umi].append(genotype)
    return cell_genotypes


def chunk(seq, num):
    """
    cut list into n chunks
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def collapse_umi(cells):
    """
    Input set of genotypes for each read
    Return list with one entry for each UMI, per cell barcode
    """
    collapsed_data = {}
    for cell_barcode, umi_set in cells.items():
        for umi, genotypes in umi_set.items():
            if len(set(genotypes)) > 1:
                pass
            else:
                try:
                    collapsed_data[cell_barcode]
                except KeyError:
                    collapsed_data[cell_barcode] = [genotypes[0]]
                else:
                    collapsed_data[cell_barcode].append(genotypes[0])
    # count total ref, total alt UMIs for each genotype
    for key, value in collapsed_data.items():
        collapsed_data[key] = [value.count("ref"), value.count("alt")]
        assert len(collapsed_data[key]) == 2
    return collapsed_data


def genotype_cells(bam, snps, cells, nproc):
    """
    Input 10x bam file and SNP coordinates with ref/alt base
    Return each cell barcode with genotype prediction
    """
    snp_set = read_snps(snps)
    if cells is not None:
        if cells.endswith(".gz"):
            known_cells = [line.strip("\n") for line in gzip.open(cells, "b")]
        else:
            known_cells = [line.strip("\n") for line in open(cells, "r")]
    else:
        known_cells = None
    p = Pool(int(nproc))
    snp_chunks = chunk(snp_set, nproc)
    data = p.map_async(functools.partial(genotype_snps,
                                   bam=bam,
                                   known_cells=known_cells),
                 snp_chunks).get(9999999)
    merged_data = merge_thread_output(data)
    return merged_data


def merge_thread_output(data):
    """
    merge multiple dictionaries of the same format into one
    """
    out = {}
    for d in data:
        for cell, counts in d.items():
            try:
                out[cell]
            except KeyError:
                out[cell] = counts
            else:
                out[cell] = [sum(x) for x in zip(counts, out[cell])]
    return out


def genotype_snps(snp_chunk, bam, known_cells):
    bamfile = pysam.AlignmentFile(bam, 'rb')
    cell_genotypes = {} # key = cell barcode, value = list of UMI-genotype pairs
    for i in snp_chunk:
        chromosome, position, ref, alt = i[0], i[1], i[2], i[3]
        # get all the reads that intersect with the snp from the bam file
        try:
            bamfile.fetch(chromosome, position, position + 1)
        except ValueError:
            pass
        else:
            for read in bamfile.fetch(chromosome, position, position + 1):
                cell_barcode, umi = scan_tags(read.tags)
                if known_cells is None or cell_barcode in known_cells:
                    genotype = get_genotype(read.query_sequence, read.pos, position, read.cigar)
                    if genotype == ref:
                        cell_genotypes = add_genotype(cell_genotypes, cell_barcode, umi, "ref")
                    elif genotype == alt:
                        cell_genotypes = add_genotype(cell_genotypes, cell_barcode, umi, "alt")
    collapsed_umi = collapse_umi(cell_genotypes)
    bamfile.close()
    if None in collapsed_umi.keys():
        collapsed_umi.pop(None)
    else:
        pass
    return collapsed_umi


def count_edit_percent_at_postion(edit_chunk, bam, known_cells):
    bamfile = pysam.AlignmentFile(bam, 'rb')
    edit_counts = {} # key = cell barcode and position, value  = UMI counts
    for i in edit_chunk:
        chromosome, position, ref, alt = i[0], i[1], i[2], i[3]
        # get all the reads that intersect with the snp from the bam file
        try:
            bamfile.fetch(chromosome, position, position + 1)
        except ValueError:
            pass
        else:
            for read in bamfile.fetch(chromosome, position, position + 1):
                cell_barcode, umi = scan_tags(read.tags)
                position_cell_index = str(cell_barcode)+":"+str(chromosome)+","+str(postion)
                if known_cells is None or cell_barcode in known_cells:
                    transcript_base = get_genotype(read.query_sequence, read.pos, position, read.cigar)
                    if transcript_base == ref:
                        edit_counts = add_genotype(edit_counts, position_cell_index, umi, "ref")
                    elif transcript_base == alt:
                        edit_counts = add_genotype(edit_counts, position_cell_index, umi, "alt")
    collapsed_umi = collapse_umi(edit_counts)
    bamfile.close()
    if None in collapsed_umi.keys():
        collapsed_umi.pop(None)
    else:
        pass
    return collapsed_umi


def save_data(data, filename):
    """
    Save table of snp counts
    """
    with open(filename, "w+") as outfile:
        outfile.write("cell_barcode\treference_count\talternate_count\n")
        for key, value in data.items():
            outfile.write(key + "\t" + str(value[0]) + "\t" + str(value[1]) + "\n")


def save_edit_data(data, filename):
    """
    Save matrix of edit counts
    """
    cell_barcodes = [i.split(':', 1)[0] for i in data.keys()]
    coordinates = [i.split(':', 1)[1] for i in data.keys()]
    # initialize new dictionary
    mat = {}
    for i in cell_barcodes:
        mat[i] = {}

    for key, value in data.items():
        cb = key.split(':')[0]
        pos = key.split(':')[1]
        mat[cb][pos] = value

    with open(filename, "w+") as outfile:
        outfile.write('postion\t'+'\t'.join(cell_barcodes)+'\n')
        for i in coordinates: # rows
            outfile.write(i)
            for j in cell_barcodes: # columns
                try:
                    mat[j][i]
                except KeyError:
                    outfile.write('\t.')
                else:
                    outfile.write('\t'+str(mat[j][i]))
            outfile.write('\n')


def edited_transcripts(bam, edit_base, cells, nproc):
    """
    Input 10x bam file and SNP coordinates with ref/alt base
    Return each cell barcode with genotype prediction
    """
    edit_set = read_snps(edit_base)
    if cells is not None:
        if cells.endswith(".gz"):
            known_cells = [line.strip("\n") for line in gzip.open(cells, "b")]
        else:
            known_cells = [line.strip("\n") for line in open(cells, "r")]
    else:
        known_cells = None
    p = Pool(int(nproc))
    edit_chunks = chunk(edit_set, nproc)
    data = p.map_async(functools.partial(count_edit_percent_at_postion,
                                   bam=bam,
                                   known_cells=known_cells),
                 edit_chunks).get(9999999)
    merged_data = merge_thread_output(data)
    return merged_data


@log_info
def countedited(options):
    """Count edited RNA bases per transcript per cell in single-cell RNA data"""
    bamfile = pysam.AlignmentFile(options.bam)
    if bamfile.has_index() is True:
        bamfile.close()
        data = edited_transcripts(options.bam, options.edit, options.cells, options.nproc)
        save_edit_data(data, options.output)
    else:
        bamfile.close()
        print("bam file not indexed")
        exit()


@log_info
def countsnps(options):
    """Count reference and alternate SNPs per cell in single-cell RNA data

    Look through a BAM file with CB and UB tags for cell barcodes and UMIs (as for
    10x Genomics single-cell RNA-seq data) and count the UMIs supporting one of two
    possible alleles at a list of known SNP positions.
    """
    bamfile = pysam.AlignmentFile(options.bam)
    if bamfile.has_index() is True:
        bamfile.close()
        data = genotype_cells(options.bam, options.snp, options.cells, options.nproc)
        save_data(data, options.output)
    else:
        bamfile.close()
        print("bam file not indexed")
        exit()


@log_info
def filterbarcodes(options):
    """Filter reads based on input list of cell barcodes

    Copy BAM entries matching a list of cell barcodes to a new BAM file.
    """
    nproc = int(options.nproc)
    # check if cell barcodes option is a file
    if os.path.isfile(options.cells):
        if options.cells.endswith(".gz"):
            cb = [line.strip("\n") for line in gzip.open(options.cells, "b")]
        else:
            cb = [line.strip("\n") for line in open(options.cells, "r")]
    else:
        cb = options.cells.split(",")

    inputBam = pysam.AlignmentFile(options.bam, 'rb')

    # get list of genomic intervals
    intervals = chunk_bam(inputBam, nproc)
    inputBam.close()

    p = Pool(nproc)

    # map chunks to cores
    tempfiles = p.map_async(functools.partial(iterate_reads,
                            bam=options.bam, sam=options.sam, output=options.output,
                            cb=cb), intervals.values()).get(9999999)

    # merge the temporary bam files
    mergestring = 'samtools merge -@ ' + str(nproc) + ' ' + options.output + ' ' + ' '.join(tempfiles)
    call(mergestring, shell=True)

    # remove temp files if merged
    if os.path.exists(options.output):
        [os.remove(i) for i in tempfiles]
    else:
        raise Exception("samtools merge failed, temp files not deleted")


@log_info
def run_genotype(options):
    """Genotype cells based on SNP counts

    Perform DBSCAN clustering to identify clusters of
    background, reference allele, alternate allele, and multiplet cells.
    """
    data = pd.read_table(options.infile)
    gt = genotype.Genotype(data)
    gt.transform_snps()
    log10_min = np.log10(float(options.min_umi))
    gt.filter_low_count(min_log10_count=log10_min)
    if options.downsample is False:
        gt.detect_background(eps=1,
                             min_samples=10000,
                             subsample=False,
                             n_jobs=options.nproc)
    else:
        gt.detect_background(eps=options.eps_background,
                             min_samples=options.min_samples_background,
                             n_jobs=options.nproc)
    gt.segment_cells()
    gt.detect_cells(eps=options.eps_cells,
                    min_samples=options.min_samples_cells,
                    n_jobs=options.nproc)
    gt.label_barcodes()
    gt.labels.to_csv(options.outfile, sep='\t', index=False)
    if options.plot is True:
        from matplotlib.pyplot import savefig
        pt = gt.plot_clusters()
        plot_name = options.infile.split('.')[0] + ".png"
        savefig(plot_name, dpi=500)
    summary_name = options.outfile.split('.')[0] + '_summary.tsv'
    gt.summarize().to_csv(summary_name, sep='\t', index=True)
