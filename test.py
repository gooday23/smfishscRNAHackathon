#!/usr/bin/env python3

# Single cell gene expression = intrinsic and extrinsic effects

import sys
import math
import os
import numpy as np
import scipy
import scipy.stats
import pandas as pd
from scipy.stats import zscore
from scipy.spatial.distance import euclidean, squareform, pdist
import smfishHmrf.reader as reader
from smfishHmrf.HMRFInstance import HMRFInstance
from smfishHmrf.DatasetMatrix import DatasetMatrix, DatasetMatrixSingleField
from smfishHmrf.bias_correction import calc_bias_moving, do_pca, plot_pca
import smfishHmrf.visualize as visualize
import smfishHmrf.spatial as spatial
from importlib.machinery import SourceFileLoader

visual = SourceFileLoader("sphx", "sphx/visual.py").load_module()


# Illumination bias correction is recommended for this data set. Illumination bias refers to biases arising from the optical instrument
# which may produce images with skewed intensity in certain regions of the imaged field.
# https://link.springer.com/protocol/10.1007%2F978-1-4939-9057-3_16

directory = "/Users/hangxu/Projects/Hackathon/rsc/smfish-hmrf/hmrf-usage/data"
genes = reader.read_genes("%s/genes" % directory)
Xcen, field = reader.read_coordinates("%s/fcortex.coordinates.txt" % directory)
expr = reader.read_expression_matrix("%s/fcortex.gene.ALL.txt" % directory)

# create a DatasetMatrixSingleField instance with the input files
# DatasetMatrixSingleField is a class that encapsulates all information about a spatial transcriptomic data set.
this_dset = DatasetMatrixSingleField(expr, genes, None, Xcen)

# construct a local neighborhood graph
# Compute the cell pairwise euclidean distance matrix using the cell coordinates in Xcen.
# Then we deTtermine a cutoff on Euclidean distance such that a pair of cells separated by less than the cut-off distance is assigned an edge.
# This cutoff can be expressed as top X-percentile of Euclidean distances.
# For example, 0.30 means a cutoff that is equal to 0.30% of all Euclidean distance values.
# In this example, we settle on 0.30% as this cutoff produces on average 5 neighbors per cell.
# Use test_adjacency_list to test a number of cut-off values.

this_dset.test_adjacency_list([0.3,0.5,1],metric="euclidean")
this_dset.calc_neighbor_graph(0.3, metric="euclidean")

# compute the independent regions of the graph
# use java script GraphColoring
this_dset.calc_independent_region()

# spatial gene selection
# pdist(X): array([ 39.16660312,  98.17094122, 488.62182718, 132.44032052, 462.98015595, 512.52675998])
# squareform(pdist(X)):
# array([[  0.        ,  39.16660312,  98.17094122, 488.62182718],
#        [ 39.16660312,   0.        , 132.44032052, 462.98015595],
#        [ 98.17094122, 132.44032052,   0.        , 512.52675998],
#        [488.62182718, 462.98015595, 512.52675998,   0.        ]])
euc = squareform(pdist(Xcen, metric="euclidean"))
dissim = spatial.rank_transform_matrix(euc, reverse=False, rbp_p=0.95)

res = spatial.calc_silhouette_per_gene(genes=genes,
                                       expr=expr,
                                       dissim=dissim,
                                       examine_top=0.1,
                                       permutation_test=True,
                                       permutations=100)
print("gene", "sil.score", "p-value")
for i,j,k in res:
    print(i,j,k)

# we select a p value cutoff in choosing spatial genes
res_df = pd.DataFrame(res, columns=["gene", "sil.score", "pval"])
res_df = res_df.set_index("gene")
new_genes = res_df[res_df.pval<=0.05].index._internal_get_values().tolist()
print(new_genes)

for g in ["calb1", "acta2", "tbr1"]:
    visual.gene_expression(this_dset, goi=g, vmax=2.0, vmin=0,
                              title=True, colormap="Reds", size_factor=5,
                              dot_size=20, outfile="%s.png" % g)

# there are other ways to select genes
# for example: use pca analysis and select genes based on which genes are correlated to top principal components from PCA analysis.
# remove cell-type variations so as to focus solely on spatial variation in the data

new_genes = reader.read_genes("%s/HMRF.genes" % directory)
new_dset = this_dset.subset_genes(new_genes)

# initiate HMRF instance
print("Running HMRF...")
outdir = "spatial.Feb20"
# Constructor of HMRFInstance requires run_name output_directory DatasetMatrix_instance K (initial_beta, beta_increment, number_of_betas)
if not os.path.isdir(outdir):
    os.mkdir(outdir)
    this_hmrf = HMRFInstance("cortex", outdir, new_dset, 9, (0, 0.5, 30), tolerance=1e-20)

# run HMRF
this_hmrf.init(nstart=1000, seed=-1)
this_hmrf.run()

# we visualize the spatial clusters in 2D
visual.domain(this_hmrf, 9, 9.0, dot_size=45, size_factor=10, outfile="visualize.beta.%.1f.png" % 9.0)

# To check if the detected spatial domains are significant,
# we compare it to a case where the spatial positions of the cells are fully shuffled (or randomly permuted).
# We first create a randomly permuted data set by shuffling the cells in the original matrix.
# The parameter 0.99 in the instance method shuffle() is the shuffling proportion.

print("Running pertubed HMRF...")
outdir = "perturbed.Feb20"
if not os.path.isdir(outdir):
os.mkdir(outdir)
perturbed_dset = new_dset.shuffle(0.999)
perturbed_hmrf = HMRFInstance("cortex", outdir, perturbed_dset, 9, (0, 0.5, 30), tolerance=1e-20)
perturbed_hmrf.init(nstart=1000, seed=-1)
perturbed_hmrf.run()

k=9
betas = np.array(range(0, 90, 5) + range(90, 150, 10)) / 10.0
lik_data, diff_data = [], []
for b in betas:
    lik_data.append((b, "observed", this_hmrf.likelihood[(k,b)]))
    lik_data.append((b, "random",perturbed_hmrf.likelihood[(k,b)]))
    diff_data.append((b, "obs - rand", this_hmrf.likelihood[(k,b)] – \
    perturbed_hmrf.likelihood[(k, b)]))
a_lik = pd.DataFrame(data={"label":[v[1] for v in lik_data], "beta":[v[0] for v in lik_data], "log-likelihood":[v[2] for v in lik_data]})
d_lik = pd.DataFrame(data={"label":[v[1] for v in diff_data], "beta":[v[0] for v in diff_data], "log-likelihood":[v[2] for v in diff_data]})
axn = sns.lmplot(x="beta", y="log-likelihood", hue="label", data=a_lik, fit_reg=False)
axn = sns.lmplot(x="beta", y="log-likelihood", hue="label", data=d_lik, fit_reg=False)