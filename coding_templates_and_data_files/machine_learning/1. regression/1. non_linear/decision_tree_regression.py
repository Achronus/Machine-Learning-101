# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('position_salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)


# Visualising the results

# Add smoother curve to graph & higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

# Create graph
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (Decision Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
