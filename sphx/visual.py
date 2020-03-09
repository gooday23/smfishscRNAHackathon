import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import sys
import re
import os
import numpy as np
import math
import pandas as pd
import seaborn as sns
from operator import itemgetter
from scipy.spatial.distance import squareform, pdist
from scipy.stats import percentileofscore
import smfishHmrf.reader as reader
from smfishHmrf.HMRFInstance import HMRFInstance
from smfishHmrf.DatasetMatrix import DatasetMatrix, DatasetMatrixSingleField, DatasetMatrixMultiField

def gene_expression(this_dataset, goi=None, foi=None, vmax=2.0, vmin=-2.0, \
                    edgecolors="none", title=True, colormap="Reds", size_factor=2, \
                    ncol=10, dot_size=20, outfile=None, xlim=(0, 100), ylim=(0, 100)):
    assert isinstance(this_dataset, DatasetMatrix), \
        "this_dataset must be of DatasetMatrixMultiField or DatasetMatrixSingleField"

    multifield = False
    if isinstance(this_dataset, DatasetMatrixMultiField):
        multiField = True

    multigene = False
    if isinstance(goi, list) and len(goi) > 1:
        multigene = True
    if len(goi) == 1:
        goi = goi[0]

    assert multifield == False or multifield == False, "multifield, multigene is not supported"

    if multifield == True and multigene == False:  # single gene, multifield
        gene_expression_multi_field(this_dataset, goi=goi, foi=foi, vmax=vmax, vmin=vmin, \
                                    edgecolors=edgecolors, title=title, colormap=colormap, size_factor=size_factor, \
                                    ncol=ncol, dot_size=dot_size, outfile=outfile)
    elif multifield == False and multigene == False:  # single gene, single field
        gene_expression_single_field(this_dataset, goi=goi, vmax=vmax, vmin=vmin, \
                                     edgecolors=edgecolors, title=title, colormap=colormap, size_factor=size_factor, \
                                     dot_size=dot_size, outfile=outfile)
    elif multifield == False and multigene == True:  # multigene single field
        multigene_expression_single_field(this_dataset, goi=goi, vmax=vmax, vmin=vmin, \
                                          edgecolors=edgecolors, title=title, colormap=colormap, ncol=ncol,
                                          size_factor=size_factor, \
                                          dot_size=dot_size, outfile=outfile, xlim=xlim, ylim=ylim)


# foi is field of interest
def gene_expression_multi_field(this_dataset, goi, foi=None, vmax=2.0, vmin=-2.0, \
                                edgecolors="none", title=True, colormap="Reds", size_factor=2, \
                                ncol=10, dot_size=20, outfile=None):
    assert isinstance(foi, (None, list)), "foi must be a list or None"
    assert isinstance(goi, str), "goi must be a str"
    assert isinstance(this_dataset, DatasetMatrixMultiField), "Dataset must be DatasetMatrixMultiField"
    X = this_dataset.expr

    Xcen = this_dataset.Xcen
    genes = this_dataset.genes
    gene_map = {}
    for ig, g in enumerate(genes):
        gene_map[g] = ig
    field = this_dataset.field

    FD = None
    if foi is None:
        FD = np.unique(field).tolist()
    else:
        FD = foi

    Xcen2 = np.copy(Xcen)
    nrow = int(len(FD) / ncol)
    if len(FD) % ncol > 0:
        nrow += 1

    f, axn = plt.subplots(nrow, ncol, figsize=(ncol * size_factor, nrow * size_factor))
    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    cm = plt.cm.get_cmap(colormap)

    ct = 0
    gid = gene_map[goi]
    for fd in FD:
        m = np.where(field == fd)[0]
        axn.flat[ct].scatter(Xcen2[m, 0], Xcen2[m, 1], s=dot_size, c=X[gid, m], edgecolors=edgecolors, \
                             cmap=cm, vmin=vmin, vmax=vmax)
        # axn.flat[ct].set_xlim(5, 30)
        # axn.flat[ct].set_ylim(5, 30)
        axn.flat[ct].title.set_visible(False)
        axn.flat[ct].set_facecolor("white")
        # plt.axis("off")
        if title:
            # axn.flat[ct].text(5, 25, "%s" % g)
            axn.flat[ct].annotate("%s (FD=%d)" % (goi, fd), (0.5, 0.95), xycoords='figure fraction', ha='center')
        ct += 1

    if outfile is None:
        plt.show()
    else:
        f.savefig(outfile)


def gene_expression_single_field(this_dataset, goi, vmax=2.0, vmin=-2.0, \
                                 edgecolors="none", title=True, colormap="Reds", size_factor=2, \
                                 dot_size=20, outfile=None):
    assert isinstance(goi, str), "gene must be str"
    assert isinstance(this_dataset, DatasetMatrixSingleField), "Dataset must be DatasetMatrixSingleField"
    X = this_dataset.expr
    Xcen = this_dataset.Xcen
    genes = this_dataset.genes
    gene_map = {}
    for ig, g in enumerate(genes):
        gene_map[g] = ig

    Xcen2 = np.copy(Xcen)

    f, axn = plt.subplots(1, 1, figsize=(size_factor, size_factor))

    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    cm = plt.cm.get_cmap(colormap)

    gid = gene_map[goi]
    axn.scatter(Xcen2[:, 0], Xcen2[:, 1], s=dot_size, c=X[gid, :], edgecolors=edgecolors, \
                cmap=cm, vmin=vmin, vmax=vmax)
    axn.title.set_visible(False)
    axn.set_facecolor("white")
    if title:
        axn.annotate("Gene %s" % goi, (0.5, 0.95), xycoords='figure fraction', ha='center')
    if outfile is None:
        plt.show()
    else:
        f.savefig(outfile)


def multigene_expression_single_field(this_dataset, goi, vmax=2.0, vmin=-2.0, \
                                      edgecolors="none", title=True, colormap="Reds", ncol=10, size_factor=2, \
                                      dot_size=20, outfile=None, xlim=(0, 100), ylim=(0, 100)):
    assert isinstance(goi, list), "goi must be a list"
    assert isinstance(this_dataset, DatasetMatrixSingleField), "Dataset must be DatasetMatrixSingleField"
    X = this_dataset.expr
    Xcen = this_dataset.Xcen
    genes = this_dataset.genes
    gene_map = {}
    for ig, g in enumerate(genes):
        gene_map[g] = ig
    Xcen2 = np.copy(Xcen)

    nrow = int(len(genes) / ncol)
    if len(genes) % ncol > 0:
        nrow += 1

    f, axn = plt.subplots(nrow, ncol, figsize=(ncol * size_factor, nrow * size_factor), \
                          sharex=True, sharey=True)

    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    cm = plt.cm.get_cmap(colormap)

    ct = 0
    for g in goi:
        gid = gene_map[g]
        axn.flat[ct].scatter(Xcen2[:, 0], Xcen2[:, 1], s=dot_size, c=X[gid, :], edgecolors=edgecolors, \
                             cmap=cm, vmin=vmin, vmax=vmax)
        axn.flat[ct].set_xlim(xlim[0], xlim[1])
        axn.flat[ct].set_ylim(ylim[0], ylim[1])
        axn.flat[ct].title.set_visible(False)
        axn.flat[ct].set_facecolor("white")
        # plt.axis("off")
        if title:
            # axn.flat[ct].text(5, 25, "%s" % g)
            axn.flat[ct].annotate("Gene %s" % g, (0.5, 0.95), xycoords='figure fraction', ha='center')
        ct += 1

    if outfile is None:
        plt.show()
    else:
        f.savefig(outfile)


def domain(this_HMRF, this_k, this_beta, dot_size=45, foi=None, xlim=(0, 100), \
           ylim=(0, 100), colormap="RdYlBu_r", size_factor=10, ncol=10, outfile=None):
    assert isinstance(this_HMRF, HMRFInstance), "this_HMRF is not of HMRFInstance"
    # assert isinstance(this_HMRF.dataset, DatasetMatrix), "the dataset in this_HMRF must be of DatasetMatrixSingleField"
    if isinstance(this_HMRF.dataset, DatasetMatrixSingleField):
        domain_single_field(this_HMRF, this_k, this_beta, dot_size=dot_size, \
                            size_factor=size_factor, colormap=colormap, outfile=outfile)
    else:
        domain_multi_field(this_HMRF, this_k, this_beta, foi=foi, dot_size=dot_size, xlim=xlim, \
                           ylim=ylim, colormap=colormap, size_factor=size_factor, ncol=ncol, outfile=outfile)


def domain_single_field(this_HMRF, this_k, this_beta, dot_size=45, \
                        size_factor=10, colormap="RdYlBu_r", outfile=None):
    assert isinstance(this_HMRF, HMRFInstance), "this_HMRF is not of HMRFInstance"
    assert isinstance(this_HMRF.dataset,
                      DatasetMatrixSingleField), "the dataset in this_HMRF must be of DatasetMatrixSingleField"
    Xcen = this_HMRF.dataset.Xcen
    f, axn = plt.subplots(1, 1, figsize=(size_factor, size_factor))
    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    cm = plt.cm.get_cmap(colormap)
    cls = this_HMRF.domain[(this_k, this_beta)]
    axn.scatter(Xcen[:, 0], Xcen[:, 1], s=dot_size, c=cls, edgecolors="gray", \
                cmap=cm, vmin=1, vmax=max(cls))
    # axn.set_xlim(xlim[0], xlim[1])
    # axn.set_ylim(ylim[0], ylim[1])
    if outfile is None:
        plt.show()
    else:
        f.savefig(outfile)


def domain_multi_field(this_HMRF, this_k, this_beta, foi=None, dot_size=45, xlim=(0, 100), \
                       ylim=(0, 100), colormap="RdYlBu_r", size_factor=10, ncol=10, outfile=None):
    assert isinstance(this_HMRF, HMRFInstance), "this_HMRF is not of HMRFInstance"
    assert isinstance(this_HMRF.dataset,
                      DatasetMatrixMultiField), "the dataset in this_HMRF must be of DatasetMatrixMultiField"
    Xcen = this_HMRF.dataset.Xcen
    field = this_HMRF.dataset.field
    FD = None
    if foi is None:
        FD = np.unique(field).tolist()
    else:
        FD = foi

    nrow = int(len(FD) / ncol)
    if len(FD) % ncol > 0:
        nrow += 1

    f, axn = plt.subplots(nrow, ncol, figsize=(ncol * size_factor, nrow * size_factor), \
                          sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    cm = plt.cm.get_cmap(colormap)
    cls = this_HMRF.domain[(this_k, this_beta)]

    ct = 0
    for fd in FD:
        m = np.where(field == fd)[0]
        axn.flat[ct].scatter(Xcen[m, 0], Xcen[m, 1], s=dot_size, c=cls[m], edgecolors=edgecolors, \
                             cmap=cm, vmin=1, vmax=max(cls))
        axn.flat[ct].set_xlim(xlim[0], xlim[1])
        axn.flat[ct].set_ylim(ylim[0], ylim[1])
        axn.flat[ct].title.set_visible(False)
        axn.flat[ct].set_facecolor("white")
        if title:
            axn.flat[ct].annotate("FD=%d" % fd, (0.5, 0.95), xycoords='figure fraction', ha='center')
        ct += 1

    if outfile is None:
        plt.show()
    else:
        f.savefig(outfile)


def domain_expression(this_HMRF, this_k, this_beta, vmin=-2.0, vmax=2.0, \
                      colormap="RdYlBu_r", outfile=None):
    # directory, roi, boi, Xcen, FDs, genes, k, field):
    assert isinstance(this_HMRF, HMRFInstance), "this_HMRF is not of HMRFInstance"
    expr = this_HMRF.dataset.expr
    cls = this_HMRF.domain[(this_k, this_beta)]
    genes = this_HMRF.dataset.genes
    gene_map = {}
    for ig, g in enumerate(genes):
        gene_map[g] = ig

    nj_union = []
    for ki in range(1, this_k + 1):
        m = np.where(cls == ki)[0]
        nx = []
        for g in genes:
            es = expr[gene_map[g], m]
            nx.append((ki, g, np.mean(es), np.std(es)))
        nx.sort(key=itemgetter(2), reverse=True)
        # nx.sort(lambda x,y:cmp(x[2], y[2]), reverse=True)
        for ni, nj, nk, nl in nx[:10]:
            nj_union = nj_union + [nj]
        for ni, nj, nk, nl in nx[-10:]:
            nj_union = nj_union + [nj]

    nj_union = set(nj_union)
    # nj_union = genes
    # nj_union = list(set(nj_union))
    cluster_expr = np.empty((len(nj_union), this_k), dtype="float32")
    nj_union_title = []
    for n in nj_union:
        nj_union_title.append(n.title())
    nt = {}
    for ki in range(1, this_k + 1):
        m = np.where(cls == ki)[0]
        nx = []
        for g in nj_union:
            es = expr[gene_map[g], m]
            nx.append((ki, g, np.mean(es), np.std(es)))
        cluster_expr[:, ki - 1] = np.array([n[2] for n in nx])
        nt[ki] = pd.Series([tx[2] for tx in nx], index=nj_union_title)

    # print(pd.DataFrame(nt))
    # f2,axn = plt.subplots(1,1, figsize=(9,20))
    # ax = sns.heatmap(pd.DataFrame(nt), annot=False, fmt=".2f", ax=axn)
    # sns.set(font_scale=1.0)
    cg = sns.clustermap(pd.DataFrame(nt), row_cluster=True, col_cluster=True, \
                        figsize=(5, 20), method="average", vmax=vmax, vmin=vmin, \
                        # col_ratios={"dendrogram":0.05}, \
                        # row_ratios={"dendrogram":0.05}, \
                        cmap=colormap)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    if outfile is None:
        plt.show()
    else:
        cg.savefig(outfile)


