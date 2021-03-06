from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


cancer_data = pd.read_csv('data/breast_cancer_diagnostic.csv')

cancer_data_clean = cancer_data.drop([23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315,
                             321, 411, 617])

obs = ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
       'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']

observed_data = cancer_data_clean.loc[:,obs]
class_data = cancer_data_clean.loc[:,'class']

labels = 'Benign', 'Malignant'

sizes = [cancer_data_clean['class'].value_counts()[2], cancer_data_clean['class'].value_counts()[4]]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True)

plt.axis('equal')
plt.show()

