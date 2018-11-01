import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats

cancer_data = pd.read_csv('C:/Users/adria/Downloads/breast_cancer_diagnostic.csv')

cancer_data_clean = cancer_data.drop([23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315,
                             321, 411, 617])
train = cancer_data_clean.sample(frac=0.7)
test = cancer_data_clean.drop(train.index)

obs = ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
       'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']

cls = ['class']

cancer_data_minus_id = cancer_data_clean.drop(columns = ['id'])

sns.pairplot(cancer_data_minus_id)
plt.show()

