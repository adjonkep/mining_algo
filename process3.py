from __future__ import division
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


cancer_data = pd.read_csv('C:/Users/adria/Downloads/breast_cancer_diagnostic.csv')

cancer_data_clean = cancer_data.drop([23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315,
                             321, 411, 617])

obs = ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
       'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']

X = cancer_data_clean.loc[:,obs]
y = cancer_data_clean.loc[:,'class']



X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
logreg

#print((sum(y_test==logreg_pred))/len(logreg_pred))
scores = cross_val_score(logreg, X, y)
print(scores.mean())
logreg_tab = confusion_matrix(y_test, logreg_pred)
print(logreg_tab)
print(metrics.classification_report(y_test, logreg_pred))