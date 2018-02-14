# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('credit_card_applications_som.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualising the results
from pylab import bone, pcolor, colorbar, plot, show
# Outputs the window of the map
bone()
# Returns all the winning nodes and takes the transpose
pcolor(som.distance_map().T)
# Adds the legend
colorbar()
# Adds markers - circle then square
markers = ['o', 's']
# Red and Green
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    # Places marker in the centre of the square
    plot(w[0] + 0.5,
         w[1] + 0.5,
         # Set the customers colour based on whether they got approval or not
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate( (mappings[(4, 2)], mappings[(5, 2)]), axis = 0 )
frauds = sc.inverse_transform(frauds)
