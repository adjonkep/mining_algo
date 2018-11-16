from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
#data = cancer_data_clean.loc[:,'uniformity_of_cell_size'].plot(kind='hist', color='yellowgreen')
sns.set_palette(sns.color_palette("BuGn_r"))
g = sns.pairplot(cancer_data_clean, vars=["uniformity_of_cell_size", "class"], diag_kind= 'kde')
plt.show()