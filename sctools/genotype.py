#! /usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, svm
from sklearn.cross_validation import train_test_split


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
            self.multiplet_rate  = None
            self.reference_count = None
            self.alternate_count = None
            self.multiplet_count = None
            self.svm_accuracy    = None
            self.snp_counts      = snp
            self.log_snps        = None
            self.barcodes        = snp.cell_barcode
            self.cells           = None
            self.background      = None
            self.multi           = None
            self.ref             = None
            self.alt             = None
            self.clusters        = None
            self.downsample_data = None
        else:
            raise(ValueError("Incorrect data structure provided, "
                             "must contain cell_barcode, reference_count, "
                             "alternate_count columns"))

    def transform_snps(self):
        """Log-transform SNP counts for reference and alternate alleles."""
        self.log_snps = self.snp_counts[['reference_count', 'alternate_count']].apply(np.log1p, 1)
        self.log_snps['cell_barcode'] = self.barcodes

    def filter_low_count(self, min_log10_count=2):
        """Remove cell barcodes with less than min_log10_count SNP counts."""
        self.filtered_cells = self.log_snps[
        (self.log_snps.reference_count > min_log10_count) & (self.log_snps.alternate_count > min_log10_count)
        ].copy()

    def detect_background(self, n=2000, eps=0.5, min_samples=100):
        """Detect background cells using dbscan clustering on a subsample of cells.
        Extrapolate labels to all cells using a support vector machine

        Parameters
        ----------
        n : int, optional
            Number of cells to select when downsampling data
        eps : float, optional
            Epsilon value for DBSCAN clustering. This is the local radius used for
            expanding clusters. The larger the value, the larger each cluster will be.
        min_samples : int, optional
            Minimum number of cells in each cluster
        """
        subsample = self.filtered_cells[['reference_count', 'alternate_count']].head(n).as_matrix()
        db_bg = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(subsample)
        train_x, test_x, train_y, test_y = train_test_split(subsample, db_bg.labels_, train_size = 0.7)
        model = svm.SVC()
        model.fit(train_x, train_y)
        self.svm_accuracy = sum(model.predict(test_x) == test_y) / len(test_y)
        self.filtered_cells['background'] = model.predict(self.filtered_cells[['reference_count', 'alternate_count']].as_matrix())
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

    def find_clusters(self, eps=0.4, min_samples=100):
        """Cluster genotypes using dbscan

        Parameters
        ----------
        eps : float, optional
            Epsilon value for DBSCAN clustering. This is the local radius used for
            expanding clusters. The larger the value, the larger each cluster will be.
        min_samples : int, optional
            Minimum number of cells in each cluster
        """
        if self.downsample_data is not None:
            cells_use = self.downsample_data
        else:
            cells_use = self.cells
        cell_data = cells_use[['reference_count', 'alternate_count']].as_matrix()
        db_cells = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(cell_data)
        self.clusters = db_cells
        # need to extrapolate clustering results if downsampled
        if self.downsample_data is not None:
            # fit svm and classify all cells
            train_x, test_x, train_y, test_y = train_test_split(cell_data,
                                                                db_cells.labels_,
                                                                train_size = 0.7)
            model = svm.SVC()
            model.fit(train_x, train_y)
            self.svm_accuracy_cells = sum(model.predict(test_x) == test_y) / len(test_y)
            self.cells['cell'] = model.predict(self.cells[['reference_count', 'alternate_count']].as_matrix())
        else:
            self.cells['cell'] = db_cells.labels_

    def label_barcodes(self):
        """Attach genotype labels to cell barcodes"""
        means = self.cells.groupby('cell').aggregate(np.mean)
        ref_cluster = means['reference_count'].argmax()
        alt_cluster = means['alternate_count'].argmax()
        multiplet_cluster = list(set([-1, 1, 0]) - set([ref_cluster, alt_cluster]))[0]

        lookup = {ref_cluster: 'ref', alt_cluster: 'alt', multiplet_cluster: 'multi'}
        col = self.cells.cell
        self.cells['label'] = [lookup[x] for x in list(col)]
        cell_data = self.cells[['cell_barcode', 'label']].copy()

        filtered_cells = list(set(self.snp_counts.cell_barcode).difference(self.filtered_cells.cell_barcode))
        bg_cells = list(self.filtered_cells[self.filtered_cells['background'] == 0].cell_barcode)
        all_bg = bg_cells + filtered_cells
        background = pd.DataFrame({'cell_barcode': pd.Series(all_bg)})
        background['label'] = 'background'

        cell_data = cell_data.append(background)
        all_data = pd.merge(self.snp_counts, cell_data, how = "inner", on = "cell_barcode")
        self.labels = all_data

    def plot_clusters(self, title = "SNP genotyping"):
        """Plot clustering results

        Parameters
        ----------
        title
            Title for the plot
        """
        if self.downsample_data is None:
            mat = self.cells.as_matrix()
        else:
            mat = self.downsample_data.as_matrix()
        core_samples_mask = np.zeros_like(self.clusters.labels_, dtype=bool)
        core_samples_mask[self.clusters.core_sample_indices_] = True
        labels = self.clusters.labels_

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = mat[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = mat[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
            plt.xlabel("Reference SNP count (log10 + 1)")
            plt.ylabel("Alternate SNP count (log10 + 1)")
            plt.title(title)
        plt.show()


def run_genotyping(data, plot=False):
    """Genotype cells based on SNP counts
    Runs all the methods in the Genotype class

    Parameters
    ----------
    data : pandas dataframe
        SNP UMI count data.
    plot : bool
        Plot clustering results

    Returns
    -------
    An object of class Genotype
    """
    gt = Genotype(data)
    gt.transform_snps()
    gt.filter_low_count()
    gt.detect_background()
    gt.find_clusters()
    gt.label_barcodes()
    if plot is True:
        gt.plot_clusters()
    return(gt)
