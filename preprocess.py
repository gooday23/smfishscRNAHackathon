#!/usr/bin/env python3

import scanpy as sc
import numpy as np

# read in counts csv
adata = sc.read_csv('/Users/hangxu/Projects/Hackathon/rsc/tasic_scRNAseq/full_scRNAseq/GSE71585_RefSeq_counts.csv',
                    delimiter=",",
                    first_column_names=bool,
                    dtype='float32')

# check dims
print(adata)

# need to transpose
tdata = sc.AnnData.transpose(adata)
print(tdata)

# Basic pre-processing
sc.pp.filter_cells(tdata, min_genes=200)
sc.pp.filter_genes(tdata, min_cells=3)

# check dims
print(tdata)

mito_genes = tdata.var_names.str.startswith('mt-')

# for each cell compute fraction of counts in mito genes vs. all genes
tdata.obs['percent_mito'] = np.sum(tdata[:,mito_genes].X, axis=1) / np.sum(tdata.X, axis=1)

# add the total counts per cell as observations-annotation to adata
tdata.obs['n_counts'] = tdata.X.sum(axis=1)

print(tdata.obs.head())
print(tdata.obs.describe())

# filtering
tdata = tdata[tdata.obs['n_genes'] < 10000, :]
tdata = tdata[tdata.obs['percent_mito'] < 0.15, :]

print(tdata)

# save filtered counts
tdata.write("cortex_filtered.h5ad", compression='gzip', compression_opts=1, force_dense=False)

# normalize counts per cell
# this is a deprecated function, use normalize_total instead
sc.pp.normalize_per_cell(tdata, counts_per_cell_after=1e4)

# logarithmize the data matrix
sc.pp.log1p(tdata)

# save normalized and log transformeed counts
tdata.write("cortex_transformed.h5ad", compression='gzip', compression_opts=1, force_dense=False)

# annotate highly_variable genes
sc.pp.highly_variable_genes(tdata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(tdata)

tdata.write("cortex_high_variance_genes.h5ad", compression='gzip', compression_opts=1, force_dense=False)

## reduce dimensions
sc.pp.pca(tdata, n_comps=50, svd_solver='arpack')
sc.pp.neighbors(tdata)

sc.tl.umap(tdata)
tdata.write("cortex_dim_reduce.h5ad", compression='gzip', compression_opts=1, force_dense=False)

# subset to highly variable genes, leave this out for full runs
tdataHighVariance = tdata[:, tdata.var['highly_variable']]
tdataHighVariance.write("cortex_dim_reduce_high_variance_subset.h5ad", compression='gzip', compression_opts=1, force_dense=False)

