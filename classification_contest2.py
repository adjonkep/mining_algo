from __future__ import division
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


train_data = pd.read_csv('C:/Users/adria/Downloads/train_data.csv')
test_data = pd.read_csv('C:/Users/adria/Downloads/test_data.csv')

train = train_data.sample(frac=0.7)
test = train_data.drop(train.index)

obs = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area_1','Wilderness_Area_2','Wilderness_Area_3','Wilderness_Area_4',
       '2702','2703','2704','2705','2706','2717','3501','3502','4201','4703','4704','4744','4758','5101','5151','6101',
       '6102','6731','7101','7102','7103','7201',
       '7202','7700','7701','7702','7709','7710','7745','7746','7755','7756','7757','7790','8703','8707',
       '8708','8771','8772','8776']

cls = ['Cover_Type']

X = train_data.loc[:,obs]
y = train_data.loc[:,'Cover_Type']


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

#print((sum(testCls==nb_pred)) / len(nb_pred))
scores = cross_val_score(logreg, X, y)
print(scores.mean())
print(metrics.classification_report(y_test, logreg_pred))