# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()

# Encoding the Independent Variable
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

# Split into separate columns
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap (Remove first column from X)
X = X[:, 1:]


# Spliting the dataset into the Training set & Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fitting the Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predictng the Test set results
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination

# Step 1
import statsmodels.formula.api as sm

# Add 50 rows of 1's at the front of the X table (Used as an intercept)
X = np.append( arr=np.ones((50, 1)).astype(int), values=X, axis=1 )

# Step 2
# Optimal Matrix and Features
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# Ordinary Least Squares Model
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()


# Step 3
# Identify the variables P-values
regressor_OLS.summary()


# Step 4
# Remove highest predictor
X_opt = X[:, [0, 1, 3, 4, 5]]

# Step 5
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# Step 3 again
regressor_OLS.summary()

#====================
# Rinse and repeat
#====================
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
