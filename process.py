from __future__ import division
import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


cancer_data = pd.read_csv('C:/Users/adria/Downloads/breast_cancer_diagnostic.csv')
#print(cancer_data.loc[:,'bare_nuclei'])
#print(cancer_data.loc[cancer_data['bare_nuclei'].isin(['?'])])
cancer_data_clean = cancer_data.drop([23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315, 321, 411, 617])
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


clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(trainObs, trainCls)
dt_pred = clf.predict(testObs)
dt_tab = confusion_matrix(testCls, dt_pred)
print(dt_tab)
#print((sum(testCls==dt_pred)) / len(dt_pred))
scores = cross_val_score(clf, X, y)
print(scores.mean())
print(metrics.classification_report(testCls, dt_pred))


#tree.export_graphviz(clf, out_file='tree.dot', feature_names= ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
       #'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses'],
       #                  class_names=['2-benign','4-malignant'])
