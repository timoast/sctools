# -*- coding: utf-8 -*-
#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
from sklearn import cluster, svm
from sklearn.model_selection import train_test_split


class Genotype:
    """Genotype each cell in single-cell RNA-seq based on relative SNP UMI counts

    Performs multiple rounds of density-based clustering to detect backgound cell barcodes,
    real single cells, and multiple cells containing the same cell barcode.

    Parameters
    ----------
    None
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
            self.snp_counts      = snp
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
            self.margin_multi    = None
        else:
            raise(ValueError("Incorrect data structure provided, "
                             "must contain cell_barcode, reference_count, "
                             "alternate_count columns"))

    def transform_snps(self):
        """Log-transform SNP counts for reference and alternate alleles."""
        self.log_snps = self.snp_counts[['reference_count', 'alternate_count']].apply(lambda x: np.log10(x+1))
        self.log_snps['cell_barcode'] = self.barcodes

    def filter_low_count(self, min_umi=10):
        """Remove cell barcodes with less than min_log10_count SNP counts

        Parameters
        ----------
        min_log10_count : int, optional
            log10(UMI) count cutoff for filtering cells

        Returns
        -------
        None
        """
        assert self.log_snps is not None, "Run genotype.transform_snps() first"
        log_count = np.log10(int(min_umi) + 1)
        self.filtered_cells = self.log_snps[
        (self.log_snps.reference_count > log_count) & (self.log_snps.alternate_count > log_count)
        ].copy()
        valid_cells = self.snp_counts.cell_barcode.isin(self.filtered_cells.cell_barcode)
        low_count = [not x for x in valid_cells]
        self.low_count = list(self.snp_counts[low_count].cell_barcode)

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
        A list of cluster labels
        """
        assert self.filtered_cells is not None, "Run genotype.filter_low_count() first"
        if (subsample is False )or (n > len(self.filtered_cells)):
            n = len(self.filtered_cells)
        cells = self.filtered_cells[['reference_count', 'alternate_count']].head(n).as_matrix()
        db_bg = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(cells)
        if (subsample is True) and (n < len(self.filtered_cells)):
            train_x, test_x, train_y, test_y = train_test_split(cells, db_bg.labels_, train_size = 0.7, test_size = 0.3)
            model = svm.SVC()
            model.fit(train_x, train_y)
            self.svm_accuracy_bg = sum(model.predict(test_x) == test_y) / len(test_y)
            clusters = self.filtered_cells['background'] = model.predict(self.filtered_cells[['reference_count', 'alternate_count']].as_matrix())
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

        Returns
        -------
        None
        """
        clusters = self.detect_background(n=n, eps=eps, min_samples=min_samples, subsample=subsample, n_jobs=n_jobs)
        bg_cells = [x == 0 for x in clusters]
        self.background_core = list(self.filtered_cells[bg_cells].cell_barcode)

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

        Returns
        -------
        None
        """
        clusters = self.detect_background(n=n, eps=eps, min_samples=min_samples, subsample=subsample, n_jobs=n_jobs)
        bg_cells = [x == 0 for x in clusters]
        cells = [x < 0 for x in clusters]
        self.background = list(self.filtered_cells[bg_cells].cell_barcode) + self.low_count
        self.cells = self.filtered_cells[cells]

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
        A dataframe containing downsampled data. This should be stored in the self.downsample_data slot, eg:
        gen = genotype.Genotype(snps)
        ...
        gen.downsample_data = gen.segment_cells()
        """
        assert self.cells is not None, "Run genotype.detect_total_background() first"

        if core_bg is False:
            bg_cells = self.filtered_cells[self.filtered_cells.cell_barcode.isin(self.background)]
            cells_use = self.cells
        else:
            assert self.background_core is not None, "Run genotype.detect_core_background() first"
            bg = list(self.filtered_cells.cell_barcode.isin(self.background_core))
            bg_cells = self.filtered_cells[bg]
            cells_use = self.filtered_cells[[not x for x in bg]]

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
                downsample_data = lower_segment.head(max_cells)
                return(downsample_data.append(upper_segment))
            else:
                downsample_data = upper_segment.head(max_cells)
                return(downsample_data.append(lower_segment))

    def detect_cell_clusters(self, eps=0.2, min_samples=100, n_jobs=1, core_bg=False):
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
        A list of cluster labels
        """
        assert self.cells is not None, "Run genotype.detect_total_background() first"
        if core_bg is True:
            assert self.background_core is not None, "Run genotype.detect_core_background() first"
            if self.downsample_data is not None:
                # need to downsample again with only the core bg cells taken out
                cells_use = self.segment_cells(core_bg=True)
            else:
                # remove core bg cells
                bg = self.filtered_cells.cell_barcode.isin(self.background_core)
                cells_use = self.filtered_cells[[not x for x in bg]].copy()
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
            clusters = model.predict(self.cells[['reference_count', 'alternate_count']].as_matrix())
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

        Returns
        -------
        None
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

    def detect_margin_cells(self, eps=0.3, min_samples=100, n_jobs=1):
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

        Returns
        -------
        None
        """
        assert self.ref_cells is not None, "Run genotype.detect_cells() first"
        clusters = self.detect_cell_clusters(eps=eps, min_samples=min_samples, n_jobs=n_jobs, core_bg=True)
        core_bg = self.filtered_cells.cell_barcode.isin(self.background_core)
        cell_data = self.filtered_cells[[not x for x in core_bg]].copy()
        cell_data['cell'] = clusters
        ref, alt, multi = cluster_labels(cell_data)
        ref_cells = list(cell_data[clusters == ref]['cell_barcode'])
        alt_cells = list(cell_data[clusters == alt]['cell_barcode'])
        multi_cells = list(cell_data[clusters == multi]['cell_barcode'])

        # intersect with barcodes labeled using smaller clustering radius
        self.margin_ref = [x for x in ref_cells if x in self.background]
        self.margin_alt = [x for x in alt_cells if x in self.background]
        self.margin_multi = [x for x in multi_cells if x in self.background]

    def label_barcodes(self):
        """Attach genotype labels to cell barcodes"""
        assert self.ref_cells is not None, "Run genotype.detect_cells() first"

        barcodes = self.ref_cells + self.alt_cells + self.multi_cells
        labels = (['ref']*len(self.ref_cells)) + (['alt']*len(self.alt_cells)) + (['multi']*len(self.multi_cells))
        cells = pd.DataFrame({'cell_barcode': pd.Series(barcodes), 'label': pd.Series(labels)})

        if self.margin_ref:  # have margin cells
            all_margin = self.margin_ref + self.margin_alt + self.margin_multi
            margin_labels = (['ref_margin']*len(self.margin_ref)) + (['alt_margin']*len(self.margin_alt)) + (['multi_margin']*len(self.margin_multi))

            bg_cells = [x for x in self.background if x not in self.background_core]
            all_bg = [x for x in bg_cells if x not in all_margin]
            bg_labels = ['background']*len(all_bg)
            core_bg_labels = ['background_core']*len(self.background_core)

            barcodes = all_margin + all_bg + self.background_core
            labels = margin_labels + bg_labels + core_bg_labels

            margin = pd.DataFrame({'cell_barcode': pd.Series(barcodes),
                                  'label': pd.Series(labels)})
            cells = cells.append(margin)

        elif self.background_core:  # have core backgound cells
            bg_cells = [x for x in self.background if x not in self.background_core]
            barcodes = self.background_core + bg_cells
            labels = (['background_core']*len(self.background_core)) + (['background']*len(bg_cells))
            bg = pd.DataFrame({'cell_barcode': pd.Series(barcodes),
                               'label': pd.Series(labels)})
            cells = cells.append(bg)

        else:  # only have backgound, ref, alt, multi
            bg = pd.DataFrame({'cell_barcode': pd.Series(self.background),
                               'label': pd.Series(['background']*len(self.background))})
            cells = cells.append(bg)

        all_data = pd.merge(self.snp_counts, cells, how = "inner", on = "cell_barcode")
        all_data['log_reference_count'] = all_data[['reference_count']].apply(lambda x: np.log10(x + 1))
        all_data['log_alternate_count'] = all_data[['alternate_count']].apply(lambda x: np.log10(x + 1))
        self.labels = all_data

    def plot_clusters(self, title = "SNP genotyping", log_scale = True):
        """Plot cell genotyping results

        Parameters
        ----------
        title : str, optional
            Title for the plot
        log_scale : bool, optional
            Plot UMI counts on a log10 scale

        Returns
        -------
        A matplotlib figure object
        """
        assert self.labels is not None, "Run genotype.label_barcodes() first"
        import matplotlib.pyplot as plt
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
        A pandas dataframe
        """
        assert self.labels is not None, "Run genotype.label_barcodes() first"
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
    reference cluster number, alternate cluster number, multiplet cluster number
    """
    multiplet_cluster = np.int64(-1)  # multiplet cluster should always be -1
    means = cell_data[cell_data.cell != -1].groupby('cell').aggregate(np.mean)
    assert len(means.index) == 2, "{} cell clusters detected (should be 2)".format(len(means.index))
    ref_cluster = means['reference_count'].argmax()
    mean_diff = abs(means['reference_count'] - means['alternate_count'])
    alt_cluster = list(set([-1, 1, 0]) - set([multiplet_cluster, ref_cluster]))[0]
    return(ref_cluster, np.int64(alt_cluster), multiplet_cluster)


def run_genotyping(data, subsample=True, basic=True):
    """Genotype cells based on SNP counts
    Wrapper for methods in the Genotype class

    Parameters
    ----------
    data : pandas dataframe
        SNP UMI count data.
    subsample : bool, optional
        Subsample cells when detecting background cluster and train
        a support vector machine to detect remaining cells.
    basic : bool, optional
        Run basic genotying without detecting cells on the border between background and real cells.

    Returns
    -------
    An object of class Genotype
    """
    gt = Genotype(data)
    gt.transform_snps()
    gt.filter_low_count()
    if basic is False:
        gt.detect_core_background()
    if subsample is False:
        gt.detect_total_background(eps=1, min_samples=10000, subsample=False)
    else:
        gt.detect_total_background()
    gt.segment_cells()
    gt.detect_cells()
    if basic is False:
        gt.detect_margin_cells()
    gt.label_barcodes()
    return(gt)
