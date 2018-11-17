print(__doc__)

from itertools import cycle
from time import time
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch

# Use all colors that matplotlib provides by default.
colors_ = cycle(colors.cnames.keys())

fig = plt.figure(figsize=(12, 4))
fig.subplots_adjust(left=0.04, right=1.5, bottom=0.1, top=0.9)

cancer_data = pd.read_csv('data/breast_cancer_diagnostic.csv')

cancer_data_clean = cancer_data.drop([23, 40, 139, 145, 158, 164, 235, 249, 275, 292, 294, 297, 315,
                             321, 411, 617])

obs = ['clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
       'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']

observed_data = cancer_data_clean.loc[:,obs]

X = observed_data.values

birch_models = [Birch(threshold=1.5, n_clusters=2),
                Birch(threshold=1.5, n_clusters=3)]
final_step = ['with 2 global clusters', 'with 3 global clusters']


for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
    t = time()
    birch_model.fit(X)
    time_ = time() - t
    print("Birch %s as the final step took %0.2f seconds" % (
          info, (time() - t)))

    # Plot result
    labels = birch_model.labels_
    centroids = birch_model.subcluster_centers_
    n_clusters = np.unique(labels).size
    print("n_clusters : %d" % n_clusters)
    silhouette_avg = silhouette_score(X, labels)

    ax = fig.add_subplot(1, 3, ind + 1)
    for this_centroid, k, col in zip(centroids, range(n_clusters), ['red', 'green', 'blue']):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1],
                   c='w', edgecolor=col, marker='.', alpha=0.5)
        if birch_model.n_clusters is None:
            ax.scatter(this_centroid[0], this_centroid[1], marker='+',
                       c='k', s=25)
    ax.set_ylim([0, 11])
    ax.set_xlim([0, 11])
    ax.set_autoscaley_on(False)
    ax.set_title('Birch %s' % info)
    silhouette_avg = silhouette_score(X, labels)
    print("The average silhouette_score is :", silhouette_avg)

plt.show()