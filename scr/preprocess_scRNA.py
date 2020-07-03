#!/usr/bin/env python

import argparse
import scanpy as sc
import numpy as np

sc.settings.autosave = True


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input', help='counts csv file')
args = parser.parse_args()

#count_csv = 'rsc/tasic_scRNAseq/full_scRNAseq/GSE71585_RefSeq_counts.csv'
count_csv = args.input

# read in counts csv

adata = sc.read_csv(count_csv,
                    delimiter=',',
                    first_column_names=bool,
                    dtype='float32')

# need to transpose
tdata = sc.AnnData.transpose(adata)

print(tdata)

# Basic pre-processing
# filter out cells have less than 200 genes expressed
sc.pp.filter_cells(tdata, min_genes=200)
print(tdata.obs['n_genes'].min())
# filter out genes expressed in less than 3 cells
sc.pp.filter_genes(tdata, min_cells=3)
print(tdata.var['n_cells'].min())

print(tdata)

# for each cell compute fraction of counts in mito genes vs. all genes
mito_genes = tdata.var_names.str.startswith('mt_')
tdata.obs['percent_mito'] = np.sum(tdata[:,mito_genes].X, axis=1) / np.sum(tdata.X, axis=1)

# add the total counts per cell as observations-annotation to adata
tdata.obs['n_counts'] = tdata.X.sum(axis=1)

print(tdata.obs.head())
print(tdata.obs.describe())

# filtering
tdata = tdata[tdata.obs['n_genes'] < 10000, :]
tdata = tdata[tdata.obs['percent_mito'] < 0.15, :]

print(tdata)

# normalize counts per cell
# after normalization, each observation has ta total count equal to 1e6 (CPM)
sc.pp.normalize_total(tdata, target_sum=1e4)

# logarithmize the data matrix
sc.pp.log1p(tdata)

# annotate highly_variable genes
sc.pp.highly_variable_genes(tdata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# plot
sc.pl.highly_variable_genes(tdata)

## reduce dimensions
sc.pp.pca(tdata, n_comps=50, svd_solver='arpack')
sc.pp.neighbors(tdata)

sc.tl.umap(tdata)
sc.pl.umap(tdata)

# subset to highly variable genes, leave this out for full runs
tdataHighVariance = tdata[:, tdata.var['highly_variable']]
tdataHighVariance.write_csvs('data/cortex_scRNA_dim_reduce_high_variance_subset')

