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
            self.svm_accuracy_bg = None
            self.svm_accuracy_cells = None
            self.snp_counts      = snp
            self.log_snps        = None
            self.barcodes        = snp.cell_barcode
            self.cells           = None
            self.clusters        = None
            self.downsample_data = None
        else:
            raise(ValueError("Incorrect data structure provided, "
                             "must contain cell_barcode, reference_count, "
                             "alternate_count columns"))

    def transform_snps(self):
        """Log-transform SNP counts for reference and alternate alleles."""
        self.log_snps = self.snp_counts[['reference_count', 'alternate_count']].apply(lambda x: np.log10(x+1))
        self.log_snps['cell_barcode'] = self.barcodes

    def filter_low_count(self, min_log10_count=.5):
        """Remove cell barcodes with less than min_log10_count SNP counts

        Parameters
        ----------
        min_log10_count : int, optional
            log10(UMI) count cutoff for filtering cells

        Returns
        -------
        None
        """
        self.filtered_cells = self.log_snps[
        (self.log_snps.reference_count > min_log10_count) & (self.log_snps.alternate_count > min_log10_count)
        ].copy()

    def detect_background(self, n=2000, eps=0.6, min_samples=300, subsample=True, n_jobs=1):
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
        None
        """
        if subsample is True:
            cells = self.filtered_cells[['reference_count', 'alternate_count']].head(n).as_matrix()
        else:
            cells = self.filtered_cells[['reference_count', 'alternate_count']].as_matrix()
        db_bg = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(cells)
        if subsample is True:
            train_x, test_x, train_y, test_y = train_test_split(cells, db_bg.labels_, train_size = 0.7, test_size = 0.3)
            model = svm.SVC()
            model.fit(train_x, train_y)
            self.svm_accuracy_bg = sum(model.predict(test_x) == test_y) / len(test_y)
            self.filtered_cells['background'] = model.predict(self.filtered_cells[['reference_count', 'alternate_count']].as_matrix())
        else:
            self.filtered_cells['background'] = db_bg.labels_
        self.cells = self.filtered_cells[self.filtered_cells.background < 0].copy()

    def segment_cells(self, cutoff=0.2):
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
        None
        """
        bg_cells = self.filtered_cells[self.filtered_cells.background == 0]
        bg_mean_ref, bg_mean_alt = np.mean(bg_cells.reference_count), np.mean(bg_cells.alternate_count)
        yintercept = bg_mean_alt / bg_mean_ref
        upper_segment = self.cells[self.cells.alternate_count > (self.cells.reference_count + yintercept)]
        lower_segment = self.cells[self.cells.alternate_count <= (self.cells.reference_count + yintercept)]
        difference = abs(len(upper_segment) - len(lower_segment))
        len_upper, len_lower = len(upper_segment), len(lower_segment)
        percent_difference = difference / max(len_upper, len_lower)
        if percent_difference > cutoff:
            max_cells = min(len_upper, len_lower)
            if len_lower > max_cells:
                downsample_data = lower_segment.head(max_cells)
                self.downsample_data = downsample_data.append(upper_segment)
            else:
                downsample_data = upper_segment.head(max_cells)
                self.downsample_data = downsample_data.append(lower_segment)

    def detect_cells(self, eps=0.4, min_samples=100, n_jobs=1):
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

        Returns
        -------
        None
        """
        if self.downsample_data is not None:
            cells_use = self.downsample_data
        else:
            cells_use = self.cells
        cell_data = cells_use[['reference_count', 'alternate_count']].as_matrix()
        db_cells = cluster.DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(cell_data)
        self.clusters = db_cells
        # need to extrapolate clustering results if downsampled
        if self.downsample_data is not None:
            # fit svm and classify all cells
            train_x, test_x, train_y, test_y = train_test_split(cell_data,
                                                                db_cells.labels_,
                                                                train_size = 0.7,
                                                                test_size = 0.3)
            model = svm.SVC()
            model.fit(train_x, train_y)
            self.svm_accuracy_cells = sum(model.predict(test_x) == test_y) / len(test_y)
            self.cells['cell'] = model.predict(self.cells[['reference_count', 'alternate_count']].as_matrix())
        else:
            self.cells['cell'] = db_cells.labels_

    def label_barcodes(self):
        """Attach genotype labels to cell barcodes"""
        means = self.cells.groupby('cell').aggregate(np.mean)
        assert len(means.index) == 3, "{} cell clusters detected (should be 2)".format(len(means.index)-1)
        ref_cluster = means['reference_count'].argmax()
        mean_diff = abs(means['reference_count'] - means['alternate_count'])
        multiplet_cluster = mean_diff[mean_diff == min(mean_diff)].index[0]
        alt_cluster = list(set([-1, 1, 0]) - set([multiplet_cluster, ref_cluster]))[0]

        lookup = {ref_cluster: 'ref', alt_cluster: 'alt', multiplet_cluster: 'multi'}
        col = self.cells.cell
        self.cells['label'] = [lookup[x] for x in list(col)]
        self.cells.loc[(self.cells['label'] == 'multi') & (self.cells['alternate_count'] < means['reference_count'][alt_cluster]), 'label'] = 'background'
        self.cells.loc[(self.cells['label'] == 'multi') & (self.cells['reference_count'] < means['alternate_count'][ref_cluster]), 'label'] = 'background'
        cell_data = self.cells[['cell_barcode', 'label']].copy()

        filtered_cells = list(set(self.snp_counts.cell_barcode).difference(self.filtered_cells.cell_barcode))
        bg_cells = list(self.filtered_cells[self.filtered_cells['background'] == 0].cell_barcode)
        all_bg = bg_cells + filtered_cells
        background = pd.DataFrame({'cell_barcode': pd.Series(all_bg)})
        background['label'] = 'background'

        cell_data = cell_data.append(background)
        all_data = pd.merge(self.snp_counts, cell_data, how = "inner", on = "cell_barcode")
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


def run_genotyping(data, subsample=True):
    """Genotype cells based on SNP counts
    Wrapper for methods in the Genotype class

    Parameters
    ----------
    data : pandas dataframe
        SNP UMI count data.
    subsample : bool, optional
        Subsample cells when detecting background cluster and train
        a support vector machine to detect remaining cells.

    Returns
    -------
    An object of class Genotype
    """
    gt = Genotype(data)
    gt.transform_snps()
    gt.filter_low_count()
    if subsample is False:
        gt.detect_background(eps=1, min_samples=10000, subsample=False)
    else:
        gt.detect_background()
    gt.segment_cells()
    gt.detect_cells()
    gt.label_barcodes()
    return(gt)
