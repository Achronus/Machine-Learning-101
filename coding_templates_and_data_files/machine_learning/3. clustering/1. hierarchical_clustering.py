# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('mall_customers.csv')
X = dataset.iloc[:,[3, 4]].values

# Use dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Fit HC to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s= 100, c = 'red', edgecolor = 'black', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s= 100, c = 'blue', edgecolor = 'black', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s= 100, c = 'green', edgecolor = 'black', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s= 100, c = 'cyan', edgecolor = 'black', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s= 100, c = 'magenta', edgecolor = 'black', label = 'Sensible')

plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
