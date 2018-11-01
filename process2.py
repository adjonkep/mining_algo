from __future__ import division
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


cancer_data = pd.read_csv('C:/Users/adria/Downloads/breast_cancer_diagnostic.csv')

cancer_data_clean = cancer_data.drop([23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315,
                             321, 411, 617])
train = cancer_data_clean.sample(frac=0.7)
test = cancer_data_clean.drop(train.index)

obs = ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
       'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']

cls = ['class']

trainObs = train.as_matrix(obs)
trainCls = train.as_matrix(cls).ravel()
testObs = test.as_matrix(obs)
testCls = test.as_matrix(cls).ravel()

#for cross-validation
obs = ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
       'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']

X = cancer_data_clean.loc[:,obs]
y = cancer_data_clean.loc[:,'class']


gnb = GaussianNB()
gnb = gnb.fit(trainObs, trainCls)
nb_pred = gnb.predict(testObs)

nb_tab = confusion_matrix(testCls, nb_pred)
print(nb_tab)

#print((sum(testCls==nb_pred)) / len(nb_pred))
scores = cross_val_score(gnb, X, y)
print(scores.mean())
print(metrics.classification_report(testCls, nb_pred))