# Recurrent Neural Networks (RNN)

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('google_stock_price_train_rnn.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling (Normalisation)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    # Creates the 60 timesteps of each value. E.g. row 1 = 0 -> 59, row 2 = 1 -> 60
    X_train.append(training_set_scaled[i-60:i, 0])
    # Contains the next value after the 60 timesteps. E.g. row 1 = last value of row 2, row 2 = last value of row 3
    # This is used to predict the next value (future value)
    y_train.append(training_set_scaled[i, 0])
# Convert to numpy array to be accepted in our RNN
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping - adding a new dimension
"""
Need to convert the array to a 3 dimension to be able to permit it into the RNN.
Keras requires 3 arguments for the input shape for this to work.

(batch_size, timesteps, input_dim)
batch_size - number of observations (1,198)
timesteps - number of time steps (60)
input_dim - number of indicators/predicters (1)

Using numpy.shape we input 0 as the first index for batch_size (outputs number of rows).
Then we use 1 as the second index for timesteps (outputs number of columns).
"""
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
"""
50 neurons in our layer, return sequences is used when having additional layers.
Input shape only needs the timesteps and input_dim as the batch_size is taken into account automatically.
"""
regressor.add(LSTM( units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) ))
# Ignore 20% of the neurons
regressor.add(Dropout(0.2))

# Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('google_stock_price_test_rnn.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# Merge the two datasets together
dataset_total = pd.concat( (dataset_train['Open'], dataset_test['Open']), axis = 0 )
# Take the last 60 stock prices and the test values
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# Reformats the data into one column
inputs = inputs.reshape(-1, 1)
# Transforms the inputs to be on the same feature scaling as the training set
inputs = sc.transform(inputs)

# Creating a data structure with 60 timesteps
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
# Reshape to a new dimension
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Identify the predicted values
predicted_stock_price = regressor.predict(X_test)
# Inverse the scaling to put them back to the normal values
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# Tuning the RNN
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(LSTM( units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) ))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
    return regressor
regressor = KerasRegressor(build_fn = build_regressor)

parameters = { 'epochs' : [100, 500], 'optimizer' : ['adam', 'rmsprop'] }
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, scoring = 'neg_mean_squared_error', cv = 2)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
