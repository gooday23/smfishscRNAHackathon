#!/usr/bin/env python3

import os
import argparse
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from yellowbrick.model_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from yellowbrick.features import Rank2D
from yellowbrick.model_selection import LearningCurve
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ROCAUC


def read_genes(fn):
    f = open(fn)
    next(f)
    genes = []
    for l in f:
        l = l.rstrip("\n")
        ll = l.split("\t")
        genes.append(ll[0])
    f.close()
    return genes

def eva_model(c, n, X, y, X_test, y_test, class_names, outdir):
    model = svm.LinearSVC(class_weight='balanced', dual=False, max_iter=10000, C=c)
    rfe = RFE(model, n_features_to_select=n)

    ## learning curve
    plt.clf()
    viz_LC = LearningCurve(
        rfe, scoring='f1_weighted', n_jobs=4
    )
    viz_LC.fit(X, y)
    viz_LC.show(outpath=outdir + '/LC.png')

    ## classification report
    plt.clf()
    viz_CR = ClassificationReport(rfe, classes=class_names, support=True)
    viz_CR.fit(X, y)
    viz_CR.score(X_test, y_test)
    viz_CR.show(outpath=outdir + '/CR.png')

    ## confusion matrix
    plt.clf()
    viz_CM = ConfusionMatrix(rfe, classes=class_names)
    viz_CM.fit(X, y)
    viz_CM.score(X_test, y_test)
    viz_CM.show(outpath=outdir + '/CM.png')

    ## precision recall curve
    plt.clf()
    viz_PRC = PrecisionRecallCurve(rfe, per_class=True, iso_f1_curves=True,
                                   fill_area=False, micro=False, classes=class_names)
    viz_PRC.fit(X, y)
    viz_PRC.score(X_test, y_test)
    viz_PRC.show(outpath=outdir + '/PRC.png',size=(1080,720))

    ## class prediction error
    plt.clf()
    viz_CPE = ClassPredictionError(
        rfe, classes=class_names
    )
    viz_CPE.fit(X, y)
    viz_CPE.score(X_test, y_test)
    viz_CPE.show(outpath=outdir + '/CPE.png')

    ## ROCAUC
    plt.clf()
    viz_RA = ROCAUC(rfe, classes=class_names, size=(1080,720))
    viz_RA.fit(X, y)
    viz_RA.score(X, y)
    viz_RA.show(outpath=outdir + '/RA.png')

    fit = rfe.fit(X,y)
    y_predict = fit.predict(X_test)
    f1 = f1_score(y_test, y_predict, average='weighted')

    features_retained_RFE = X.columns[rfe.get_support()].values
    feature_df =pd.DataFrame(features_retained_RFE.tolist())
    feature_df.to_csv(outdir + '/features.csv', sep='\t', index=False)

    return f1

#### files ######


# # high variable genes
# genes_fn = 'rsc/highly_variable_genes.tsv'

# # scRNA
# # txt file of normalized scRNAseq data for 113 genes x 1723 cells
# scrna_data_fn = 'rsc/tasic_scRNAseq/tasic_training_b2.txt'

# # tsv file of cell type labels for scRNAseq
# scrna_label_fn = 'rsc/tasic_scRNAseq/tasic_labels.tsv'

# # seqFish
# # txt file of normalized seqFISH data for 113 genes x 1597 cells
# seqfish_data_fn = 'rsc/tasic_scRNAseq/seqfish_cortex_b2_testing.txt'

# # tsv file of spatial cluster labels and SVM learned cell types for seqFISH
# seqfish_label_fn = 'rsc/tasic_scRNAseq/seqfish_labels.tsv'

######

parser = argparse.ArgumentParser()
parser.add_argument('-g', dest='genes', help='txt file of high variable genes')
parser.add_argument('-r', dest='scrna', help='txt file of normalized scRNAseq data')
parser.add_argument('-rl', dest='scrnalabel', help='tsv file of cell type labels for scRNAseq')
parser.add_argument('-s', dest='seqfish', help='txt file of normalized seqFISH data')
parser.add_argument('-sl', dest='seqfishlabel', help='tsv file of spatial cluster labels and SVM learned cell types for seqFISH')

args = parser.parse_args()
genes_fn = args.genes
scrna_data_fn = args.scrna
scrna_label_fn = args.scrnalabel
seqfish_data_fn = args.seqfish
seqfish_label_fn = args.seqfishlabel

######### read files ###########

var_genes = read_genes(genes_fn)

scrna_expr = pd.read_csv(scrna_data_fn, sep='\t', header=None, index_col=0)
scrna_label = pd.read_csv(scrna_label_fn, sep='\t', header=None)

seqfish_expr = pd.read_csv(seqfish_data_fn, sep='\t', header=None, index_col=0)
seqfish_label = pd.read_csv(seqfish_label_fn, sep='\t', header=None)


### filter to high variable genes ###
X = scrna_expr.T.reset_index()[var_genes]
y = scrna_label[0]
X_test = seqfish_expr.T.reset_index()[var_genes]
y_test = seqfish_label[2]

### models

class_names = ['Astrocyte',
               'Endothelial Cell',
               'GABA-ergic Neuron',
               'Glutamatergic Neuron',
               'Microglia',
               'Oligodendrocyte.1',
               'Oligodendrocyte.2',
               'Oligodendrocyte.3']


# c_candidates = np.logspace(-6,0,num=0+6+1,base=10).tolist()
# n_candidates = np.arange(7,44)
# c_list = []
# n_list = []
# f1 = []

# for c in c_candidates:
#     for n in n_candidates:
#         this_out = 'figures/eva/c' + str(c) + '_n' + str(n)
#         os.makedirs(this_out,exist_ok=True)
#         c_list.append(c)
#         n_list.append(n)
#         f1.append(eva_model(c=c, n=n, outdir=this_out, X=X, y=y, X_test=X_test, y_test=y_test,
#                             class_names=class_names))


# df = pd.DataFrame(list(zip(c_list, n_list, f1)),
#                   columns=['C','Num_features','f1_score'])

# df.to_csv('data/seqfish_evamodel_f1.tsv', sep='\t', index=False)


# def check_model(c, n, X, y, X_test, y_test, class_names, outdir):
#     model = svm.LinearSVC(class_weight='balanced', dual=False, max_iter=10000, C=c)
#     rfe = RFE(model, n_features_to_select=n)

#     fit = rfe.fit(X,y)
#     y_predict = fit.predict(X_test)
#     predict_df = pd.DataFrame(y_predict.tolist())
#     predict_df.to_csv(outdir + '/predict_label.csv', sep='\t', index=False)



# test = [[1e-3,12],[1e-6,7],[1.0,32],[1e-6,27]]
# for c,n in test:
#     print(str(c)+ '-' +str(n))
#     this_out = 'figures/eva/c' + str(c) + '_n' + str(n)
#     check_model(c=c, n=n, outdir=this_out, X=X, y=y, X_test=X_test, y_test=y_test,
#               class_names=class_names)


## plot RFECV for LinearSVC
for c in [1e-6, 1e-3, 1]:
    model = svm.LinearSVC(class_weight='balanced', dual=False, max_iter=10000, C=c)
    viz = RFECV(model, scoring='f1_weighted')
    viz.fit(X, y)
    viz.show(outpath='figures/linear_svc_rfecv.pdf')



