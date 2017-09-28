# -*- coding: utf-8 -*-
#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
from sklearn import cluster, svm
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Genotype:
    """Genotype each cell in single-cell RNA-seq based on relative SNP UMI counts

    Performs multiple rounds of density-based clustering to detect backgound cell barcodes,
    real single cells, and multiple cells containing the same cell barcode.
    """

    def __init__(self, snp):
        """Initialize the Genotype object

        Parameters
        ----------
        snp
            Pandas dataframe with columns 'cell_barcode', 'reference_count', 'alternate_count'

        Raises
        ------
        ValueError
            If `snp` does not have correct column names
        """
        if set(['cell_barcode', 'reference_count', 'alternate_count']).issubset(snp.columns):
            self.obs_multiplet_rate = None
            self.estimated_multiplet_rate = None
            self.reference_count = None
            self.alternate_count = None
            self.multiplet_count = None
            self.low_count       = None
            self.svm_accuracy_bg = None
            self.svm_accuracy_cells = None
            self.snp_counts      = snp.copy()
            self.log_snps        = None
            self.barcodes        = snp.cell_barcode
            self.cells           = None
            self.downsample_data = None
            self.background_core = None
            self.background      = None
            self.ref_cells       = None
            self.alt_cells       = None
            self.multi_cells     = None
            self.margin_ref      = None
            self.margin_alt      = None
            self.labels          = snp.copy()

            self.labels['label'] = "Unknown"
            self.labels['log_reference_count'] = self.labels['reference_count'].apply(lambda x: np.log10(x+1))
            self.labels['log_alternate_count'] = self.labels['alternate_count'].apply(lambda x: np.log10(x+1))
        else:
            raise(ValueError("Incorrect data structure provided, "
                             "must contain cell_barcode, reference_count, "
                             "alternate_count columns"))

    def reset(self):
        """Reset object back to the initial data
        """
        self.labels = self.snp_counts
        self.labels['label'] = "Unknown"
        self.labels['log_reference_count'] = self.labels['reference_count'].apply(lambda x: np.log10(x+1))
        self.labels['log_alternate_count'] = self.labels['alternate_count'].apply(lambda x: np.log10(x+1))

    def transform_snps(self):
        """Log-transform SNP counts for reference and alternate alleles
        Applies log10(count+1) to each SNP UMI count entry
        """
        self.log_snps = self.filtered_cells[['reference_count', 'alternate_count']].apply(lambda x: np.log10(x+1))
        self.log_snps['cell_barcode'] = self.filtered_cells.cell_barcode

    def filter_low_count(self, min_umi_total=20, min_umi_each=10):
        """Remove cell barcodes with low SNP counts

        Parameters
        ----------
        min_umi_total : int, optional
            Combined UMI count cutoff for filtering cells. Default is 20.
        min_umi_each : int, optional
            Minimum UMI count for each genotype. Default is 10
        """
        min_umi_total = int(min_umi_total)
        min_umi_each = int(min_umi_each)

        ok_total = (self.snp_counts.reference_count + self.snp_counts.alternate_count) > min_umi_total
        ok_each = (self.snp_counts.reference_count > min_umi_each) & (self.snp_counts.alternate_count > min_umi_each)
        self.filtered_cells = self.snp_counts[ok_total & ok_each].copy()
        valid_cells = self.snp_counts.cell_barcode.isin(self.filtered_cells.cell_barcode)
        low_count = [not x for x in valid_cells]
        self.low_count = list(self.snp_counts[low_count].cell_barcode)
        self.labels = self.labels[valid_cells]

    def detect_background(self, n=2000, eps=0.5, min_samples=300, subsample=True, n_jobs=1):
        """Detect background cells using dbscan clustering
        Extrapolate labels to all cells using a support vector machine
        if cells are first downsampled.

        Parameters
        ----------
        n : int, optional
            Number of cells to select when downsampling data
        eps : float, optional
            Epsilon value for DBSCAN clustering. This is the local radius used for
            expanding clusters. The larger the value, the larger each cluster will be.
        min_samples : int, optional
            Minimum number of cells in each cluster
        subsample : Bool, optional
            Subsample cells before detecting background cluster (much faster).
            Default is to subsample to `n` cells and then extrapolate clustering information
            to remaining cells by training a model. If setting subsample to False, the parameters
            `min_sample` and `eps` should be set much higher (eg 10,000 and 1)
        n_jobs : int, optional
            Number of cores to use for dbscan clustering. Default is 1. Setting to -1 will use all cores.
            Should only be needed when setting `subsample` to False.

        Returns
        -------
        list
            A list of cluster labels
        """
        assert self.log_snps is not None, "Run genotype.transform_snps() first"
        if (subsample is False )or (n > len(self.log_snps)):
            n = len(self.log_snps)
        cells = self.log_snps[['reference_count', 'alternate_count']].sample(n).as_matrix()
        db_bg = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(cells)
        if (subsample is True) and (n < len(self.log_snps)):
            train_x, test_x, train_y, test_y = train_test_split(cells, db_bg.labels_, train_size = 0.7, test_size = 0.3)
            model = svm.SVC()
            model.fit(train_x, train_y)
            self.svm_accuracy_bg = sum(model.predict(test_x) == test_y) / len(test_y)
            clusters = model.predict(self.log_snps[['reference_count', 'alternate_count']].as_matrix())
            return(list(clusters))
        else:
            return(list(db_bg.labels_))

    def detect_core_background(self, n=2000, eps=0.3, min_samples=300, subsample=True, n_jobs=1):
        """Detect core background cells using dbscan clustering
        Extrapolate labels to all cells using a support vector machine
        if cells are first downsampled.

        Sets self.background_core to list of cell barcodes

        Parameters
        ----------
        n : int, optional
            Number of cells to select when downsampling data
        eps : float, optional
            Epsilon value for DBSCAN clustering. This is the local radius used for
            expanding clusters. The larger the value, the larger each cluster will be.
        min_samples : int, optional
            Minimum number of cells in each cluster
        subsample : Bool, optional
            Subsample cells before detecting background cluster (much faster).
            Default is to subsample to `n` cells and then extrapolate clustering information
            to remaining cells by training a model. If setting subsample to False, the parameters
            `min_sample` and `eps` should be set much higher (eg 10,000 and 1)
        n_jobs : int, optional
            Number of cores to use for dbscan clustering. Default is 1. Setting to -1 will use all cores.
            Should only be needed when setting `subsample` to False.
        """
        clusters = self.detect_background(n=n, eps=eps, min_samples=min_samples, subsample=subsample, n_jobs=n_jobs)
        bg_cells = [x == 0 for x in clusters]
        self.background_core = list(self.log_snps[bg_cells].cell_barcode)
        self.labels.loc[(self.labels.cell_barcode.isin(self.background_core)), 'label'] = 'background_core'

    def detect_total_background(self, n=2000, eps=0.5, min_samples=300, subsample=True, n_jobs=1):
        """Detect all background cells using dbscan clustering
        Extrapolate labels to all cells using a support vector machine
        if cells are first downsampled.

        Sets self.background to list of cell barcodes
        Sets self.cells to pandas dataframe

        Parameters
        ----------
        n : int, optional
            Number of cells to select when downsampling data
        eps : float, optional
            Epsilon value for DBSCAN clustering. This is the local radius used for
            expanding clusters. The larger the value, the larger each cluster will be.
        min_samples : int, optional
            Minimum number of cells in each cluster
        subsample : Bool, optional
            Subsample cells before detecting background cluster (much faster).
            Default is to subsample to `n` cells and then extrapolate clustering information
            to remaining cells by training a model. If setting subsample to False, the parameters
            `min_sample` and `eps` should be set much higher (eg 10,000 and 1)
        n_jobs : int, optional
            Number of cores to use for dbscan clustering. Default is 1. Setting to -1 will use all cores.
            Should only be needed when setting `subsample` to False.
        """
        clusters = self.detect_background(n=n, eps=eps, min_samples=min_samples, subsample=subsample, n_jobs=n_jobs)
        bg_cells = [x == 0 for x in clusters]
        cells = [x < 0 for x in clusters]
        self.background = list(self.log_snps[bg_cells].cell_barcode) + self.low_count
        self.cells = self.log_snps[cells]
        bg_cell = self.labels.cell_barcode.isin(self.background)
        not_core = self.labels.label != 'background_core'
        self.labels.loc[(bg_cell) & (not_core), 'label'] = 'background'
        self.labels.loc[(self.labels.cell_barcode.isin(self.cells.cell_barcode)), 'label'] = 'cell'

    def segment_cells(self, cutoff=0.2, core_bg=False):
        """Segment cell population into two halves and assess density distribution

        Draw 45º line from background cluster center, then count number of remaining cells
        on each side of the line. If there is greater than `cutoff` difference in cell count,
        subsample cells in the larger segment to the number in the smaller segment.

        Parameters
        ----------
        cutoff : float, optional
            Largest allowed percentage difference in cell counts between the two segments
            before the cells in the larger segment will be downsampled to match the smaller segment.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing downsampled data. This should be stored in the self.downsample_data slot, eg:
            gen = genotype.Genotype(snps)
            ...
            gen.downsample_data = gen.segment_cells()
        """
        assert self.cells is not None, "Run genotype.detect_total_background() first"

        if core_bg is False:
            bg_cells = self.log_snps[self.log_snps.cell_barcode.isin(self.background)]
            cells_use = self.cells
        else:
            assert self.background_core is not None, "Run genotype.detect_core_background() first"
            bg = self.log_snps.cell_barcode.isin(self.background_core)
            bg_cells = self.log_snps[bg]
            cells_use = self.log_snps[[not x for x in bg]].copy()

        bg_mean_ref, bg_mean_alt = np.mean(bg_cells.reference_count), np.mean(bg_cells.alternate_count)
        yintercept = bg_mean_alt / bg_mean_ref
        upper_segment = cells_use[cells_use.alternate_count > (cells_use.reference_count + yintercept)]
        lower_segment = cells_use[cells_use.alternate_count <= (cells_use.reference_count + yintercept)]
        difference = abs(len(upper_segment) - len(lower_segment))
        len_upper, len_lower = len(upper_segment), len(lower_segment)
        percent_difference = difference / max(len_upper, len_lower)
        if percent_difference > cutoff:
            max_cells = min(len_upper, len_lower)
            if len_lower > max_cells:
                downsample_data = lower_segment.sample(max_cells)
                return(downsample_data.append(upper_segment))
            else:
                downsample_data = upper_segment.sample(max_cells)
                return(downsample_data.append(lower_segment))
        else:
            return(None)

    def detect_cell_clusters(self, eps=0.2, min_samples=100, n_jobs=1, core_bg=False, force_full=False):
        """Cluster genotypes using dbscan

        Parameters
        ----------
        eps : float, optional
            Epsilon value for DBSCAN clustering. This is the local radius used for
            expanding clusters. The larger the value, the larger each cluster will be.
        min_samples : int, optional
            Minimum number of cells in each cluster
        n_jobs : int, optional
            Number of cores to use for dbscan clustering. Default is 1. Setting to -1 will use all cores.
            Should only be needed when setting `subsample` to False.
        force_full : bool, optional
            Do not downsample data

        Returns
        -------
        list
            A list of cluster labels
        """
        assert self.cells is not None, "Run genotype.detect_total_background() first"
        if core_bg:
            assert self.background_core is not None, "Run genotype.detect_core_background() first"
            if self.downsample_data is not None:
                # need to downsample again with only the core bg cells taken out
                cells_use = self.segment_cells(core_bg=True)
            else:
                # remove core bg cells
                bg = self.log_snps.cell_barcode.isin(self.background_core)
                cells_use = self.log_snps[[not x for x in bg]].copy()
        elif self.downsample_data is not None:
            cells_use = self.downsample_data.copy()
        else:
            cells_use = self.cells.copy()
        cell_data = cells_use[['reference_count', 'alternate_count']].as_matrix()
        db_cells = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(cell_data)
        # need to extrapolate clustering results if downsampled
        if (self.downsample_data is not None) and (force_full is False):
            # fit svm and classify all cells
            train_x, test_x, train_y, test_y = train_test_split(cell_data,
                                                                db_cells.labels_,
                                                                train_size = 0.7,
                                                                test_size = 0.3)
            model = svm.SVC()
            model.fit(train_x, train_y)
            self.svm_accuracy_cells = sum(model.predict(test_x) == test_y) / len(test_y)
            # run on full dataset
            if core_bg:
                bg = self.log_snps.cell_barcode.isin(self.background_core)
                cells = self.log_snps[[not x for x in bg]]
                cells = cells[['reference_count', 'alternate_count']].as_matrix()
            else:
                cells = self.cells[['reference_count', 'alternate_count']].as_matrix()
            clusters = model.predict(cells)
            return(list(clusters))
        else:
            return(list(db_cells.labels_))

    def detect_cells(self, eps=0.2, min_samples=100, n_jobs=1):
        """Detect cell clusters

        Parameters
        ----------
        eps : float, optional
            Epsilon value for DBSCAN clustering. This is the local radius used for
            expanding clusters. The larger the value, the larger each cluster will be.
        min_samples : int, optional
            Minimum number of cells in each cluster
        n_jobs : int, optional
            Number of cores to use for dbscan clustering. Default is 1. Setting to -1 will use all cores.
            Should only be needed when setting `subsample` to False.
        """
        clusters = self.detect_cell_clusters(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
        cell_data = self.cells.copy()
        cell_data['cell'] = clusters
        ref, alt, multi = cluster_labels(cell_data)

        lookup = {ref: 'ref', alt: 'alt', multi: 'multi'}
        col = cell_data.cell
        cell_data['label'] = [lookup[x] for x in list(col)]
        means = cell_data.groupby('cell').aggregate(np.mean)
        cell_data.loc[(cell_data['label'] == 'multi') & (cell_data['alternate_count'] \
                        < means['reference_count'][alt]), 'label'] = 'background'
        cell_data.loc[(cell_data['label'] == 'multi') & (cell_data['reference_count'] \
                        < means['alternate_count'][ref]), 'label'] = 'background'

        self.ref_cells = list(cell_data[cell_data.label == 'ref']['cell_barcode'])
        self.alt_cells = list(cell_data[cell_data.label == 'alt']['cell_barcode'])
        self.multi_cells = list(cell_data[cell_data.label == 'multi']['cell_barcode'])

        self.labels.loc[(self.labels.cell_barcode.isin(self.ref_cells)), 'label'] = 'ref'
        self.labels.loc[(self.labels.cell_barcode.isin(self.alt_cells)), 'label'] = 'alt'
        self.labels.loc[(self.labels.cell_barcode.isin(self.multi_cells)), 'label'] = 'multi'

    def detect_margin_cells(self, eps=0.2, min_samples=100, n_jobs=1):
        """Detect cells on the margin between true cells and background droplets

        This should be run with a larger DBSCAN epsilon value than
        used for the detect_cells() function.

        Parameters
        ----------
        eps : float, optional
            Epsilon value for DBSCAN clustering. This is the local radius used for
            expanding clusters. The larger the value, the larger each cluster will be.
        min_samples : int, optional
            Minimum number of cells in each cluster
        n_jobs : int, optional
            Number of cores to use for dbscan clustering. Default is 1. Setting to -1 will use all cores.
            Should only be needed when setting `subsample` to False.
        """
        assert self.ref_cells is not None, "Run genotype.detect_cells() first"
        clusters = self.detect_cell_clusters(eps=eps, min_samples=min_samples, n_jobs=n_jobs, core_bg=True)
        core_bg = self.log_snps.cell_barcode.isin(self.background_core)
        cell_data = self.log_snps[[not x for x in core_bg]].copy()
        cell_data['cell'] = clusters
        ref, alt, _ = cluster_labels(cell_data)
        ref_cells = list(cell_data[clusters == ref]['cell_barcode'])
        alt_cells = list(cell_data[clusters == alt]['cell_barcode'])

        # intersect with barcodes labeled using smaller clustering radius
        self.margin_ref = list(set(ref_cells) & set(self.background))
        self.margin_alt = list(set(alt_cells) & set(self.background))
        self.labels.loc[(self.labels.cell_barcode.isin(self.margin_ref)), 'label'] = 'ref_margin'
        self.labels.loc[(self.labels.cell_barcode.isin(self.margin_alt)), 'label'] = 'alt_margin'

    def plot(self, title = "SNP genotyping", log_scale = True):
        """Plot cell genotyping results

        Parameters
        ----------
        title : str, optional
            Title for the plot
        log_scale : bool, optional
            Plot UMI counts on a log10 scale

        Returns
        -------
        figure
            A matplotlib figure object
        """
        groups = self.labels.groupby('label')
        fig, ax = plt.subplots()
        if log_scale:
            for name, group in groups:
                ax.plot(group.log_reference_count, group.log_alternate_count,
                        marker='.', linestyle='', ms=2, label=name, alpha=0.5)
            ax.legend()
            ax.set_ylabel("Alternate UMI counts (log10 + 1)")
            ax.set_xlabel("Reference UMI counts (log10 + 1)")
        else:
            for name, group in groups:
                ax.plot(group.reference_count, group.alternate_count,
                        marker='.', linestyle='', ms=2, label=name, alpha=0.5)
            ax.legend()
            ax.set_ylabel("Alternate UMI counts")
            ax.set_xlabel("Reference UMI counts")
        ax.set_title(title)
        return(fig)

    def summarize(self):
        """Count number of cells of each genotype and
        estimate the rate of cell multiplets

        Parameters
        ----------
        None

        Returns
        -------
        pandas.DataFrame
            A pandas dataframe
        """
        self.multiplet_count = sum(self.labels.label == 'multi')
        self.reference_count = sum(self.labels.label == 'ref')
        self.alternate_count = sum(self.labels.label == 'alt')
        self.background_count = sum(self.labels.label == 'backgound')
        self.total_cells = self.reference_count + self.alternate_count + self.multiplet_count
        self.obs_multi_rate = self.multiplet_count / self.total_cells
        self.estimated_multiplet_rate = self.obs_multi_rate / (min(self.reference_count, self.alternate_count) / (self.reference_count + self.alternate_count))
        dat = pd.DataFrame({'Count': [self.reference_count, self.alternate_count,
                                      self.multiplet_count, self.estimated_multiplet_rate*self.total_cells],
                            'Percentage': [self.reference_count / self.total_cells, self.alternate_count / self.total_cells,
                                           self.obs_multi_rate, self.estimated_multiplet_rate]},
                            index = ['Reference', 'Alternate', 'Observed Multiplet', 'Estimated Multiplet'])
        return(dat)


def cluster_labels(cell_data):
    """Label cluster numbers as ref, alt, multi based on SNP UMI count means

    Parameters
    ----------
    None

    Returns
    -------
    tuple
        reference cluster number, alternate cluster number, multiplet cluster number
    """
    multiplet_cluster = np.int64(-1)  # multiplet cluster should always be -1
    means = cell_data[cell_data.cell != -1].groupby('cell').aggregate(np.mean)
    assert len(means.index) == 2, "{} cell clusters detected (should be 2)".format(len(means.index))
    ref_cluster = means['reference_count'].argmax()
    mean_diff = abs(means['reference_count'] - means['alternate_count'])
    alt_cluster = list(set([-1, 1, 0]) - set([multiplet_cluster, ref_cluster]))[0]
    return(ref_cluster, np.int64(alt_cluster), multiplet_cluster)


def run_genotyping(data, min_umi_total=20, min_umi_each=10, subsample=True, margin=False, nproc=1,
                   eps_background=0.5, eps_background_core=0.2, eps_cells=0.2, eps_margin=0.1,
                   min_drops_background=300, min_drops_cells=100, max_difference=0.2):
    """Genotype cells based on SNP counts

    Performs iterative density-based clustering using the DBSCAN algorithm to detect
    background cluster of empty droplets, a cluster of cells for each genotype or species,
    and a cluster of multiplet cells.

    Parameters
    ----------
    data : pandas dataframe
        SNP UMI count data.
    min_umi_total : int, optional
        Combined UMI count cutoff for filtering cells. Default is 20.
    min_umi_each : int, optional
        Minimum UMI count for each genotype. Default is 10
    subsample : bool, optional
        Subsample cells when detecting background cluster and train
        a support vector machine to detect remaining cells. Default is True.
    margin : bool, optional
        Detect cells on the border between background and real cells. Default is False.
    nproc : int, optional
        Number of processors. Default is 1, setting to -1 will use all cores.
    max_difference : float, optional
        Maximum UMI count difference between genotypes before applying downsampling to equalize UMI count distribution. Default is 0.2
    eps_background : float, optional
        Epsilon value passed to DBSCAN for detection of background cluster. Default is 0.5.
    eps_background_core : float, optional
        Epsilon value passed to DBSCAN for detection of core background cluster. Default is 0.2.
    eps_cells : float, optional
        Epsilon value passed to DBSCAN for detection of cell clusters. Default is 0.2.
    eps_margin : float, optional
        Epsilon value passed to DBSCAN for detection of margin cells. Default is 0.1.
    min_drops_background : int, optional
        Minimum number of barcodes per cluster allowed during detection of background cluster.
        Default is 300.
    min_drops_cells : int, optional
        Minimum number of barcodes per cluster allowed during detection of cell clusters.
        Default is 100.

    Returns
    -------
    Genotype
        An object of class Genotype
    """
    gt = Genotype(data)
    gt.filter_low_count(min_umi_total=min_umi_total, min_umi_each=min_umi_each)
    gt.transform_snps()
    if margin:
        gt.detect_core_background(subsample=subsample, eps=eps_background_core)
    gt.detect_total_background(eps=eps_background, min_samples=min_drops_background,
                               n_jobs=nproc)
    gt.downsample_data = gt.segment_cells(cutoff=max_difference)
    gt.detect_cells(eps=eps_cells,
                    min_samples=min_drops_cells,
                    n_jobs=nproc)
    if margin:
        gt.detect_margin_cells(eps=eps_margin)
    return(gt)
