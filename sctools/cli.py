# -*- coding: utf-8 -*-
#! /usr/bin/env python

from __future__ import absolute_import
from sctools import sctools, genotype
import pandas as pd
import functools
import time


def log_info(func):
    """Decorator that prints function arguments and runtime
    """
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


@log_info
def run_filterbarcodes(options):
    """Wraps the sctools.filterbarcodes function for use on the command line
    """
    sctools.filterbarcodes(cells=options.cells, bam=options.bam,
                           output=options.output, sam=options.sam, nproc=options.nproc)


@log_info
def run_genotyping(options):
    """Wraps the genotype.run_genotyping function for use on the command line
    """
    data = pd.read_table(options.infile)
    gt = genotype.run_genotyping(data=data,
                                 min_umi_total=options.min_umi_total,
                                 min_umi_each=options.min_umi_each,
                                 subsample=options.downsample,
                                 margin=options.margin,
                                 max_difference=options.max_difference,
                                 eps_background=options.eps_background,
                                 eps_background_core=options.eps_background_core,
                                 eps_cells=options.eps_cells,
                                 eps_margin=options.eps_margin,
                                 min_drops_background=options.min_samples_background,
                                 min_drops_cells=options.min_samples_cells)
    if options.plot:
        import matplotlib.pyplot as plt
        pt = gt.plot()
        plot_name = options.sample_name + ".png"
        plt.savefig(plot_name, dpi=500)
        plt.close(pt)
    if options.summarize:
        summary = gt.summarize()
        summary.to_csv(options.sample_name + "_summary.tsv", sep="\t", index=False)
    gt.labels[['cell_barcode', 'reference_count',
               'alternate_count', 'label']].to_csv(options.sample_name + "_genotypes.tsv",
                                                   sep='\t', index=False)


@log_info
def run_countsnps(options):
    """Wraps the sctools.countsnps function for use on the command line
    """
    data = sctools.countsnps(bam=options.bam, snp=options.snp, cells=options.cells, nproc=options.nproc)
    sctools.save_data(data, options.output)

@log_info
def run_countedited(options):
    """Wraps the sctools.countedited function for use on the command line
    """
    data = sctools.countedited(bam=options.bam, edit=options.edit, cells=options.cells, nproc=options.nproc)
    sctools.save_edit_data(data, options.output)
