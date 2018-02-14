# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('salary_data.csv')

# YearsExperience Column
X = dataset.iloc[:, :-1].values

# Salary Column
y = dataset.iloc[:, 1].values

# Spliting the dataset into the Training set & Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#-----------------------------------------------

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

# Create the SLR machine/model
regressor = LinearRegression()

# Fit the training data to the machine/model so that it
# learns the correlations of the YearsExperience and Salary
regressor.fit(X_train, y_train)

#-----------------------------------------------

# Predicting the Test set results

# Vector of predictions of the dependent variable
# Contains the predicted Salaries for all observations of the test set
y_pred = regressor.predict(X_test)

#-----------------------------------------------

# Visualising the Training set results

# Plotting the provided data
plt.scatter(X_train, y_train, color="red")

# Plotting predictions
plt.plot(X_train, regressor.predict(X_train), color="blue")

# Assign core values to graph
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show() # Specifies end of graph and to display it


# Visualising the Test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
