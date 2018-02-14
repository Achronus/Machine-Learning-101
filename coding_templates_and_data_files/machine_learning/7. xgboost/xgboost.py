# XGBoost

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('churn_modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X1 = LabelEncoder()
labelEncoder_X2 = LabelEncoder()
X[:, 1] = labelEncoder_X1.fit_transform(X[:, 1])
X[:, 2] = labelEncoder_X2.fit_transform(X[:, 2])

oneHotEncoder = OneHotEncoder(categorical_features=[1])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]

# Spliting the dataset into the Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
