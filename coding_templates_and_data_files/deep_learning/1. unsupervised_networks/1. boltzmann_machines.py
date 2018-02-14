# Boltzmann Machines
"""
This example is used to create a Recommender System that predicts binary ratings 'Like' or 'Not Liked'
"""

# Importing the Libraries
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int64')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int64')

# Getting the number of users and movies
nb_users = int(max( max(training_set[:, 0]), max(test_set[:, 0]) ))
nb_movies = int(max( max(training_set[:, 1]), max(test_set[:, 1]) ))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    # nv = visible nodes
    # nh = hidden nodes
    def __init__(self, nv, nh):
        # Initialize the weights - this consists of a matrix with the size of the hidden nodes and visible nodes
        self.W = torch.randn(nh, nv)
        # Initialize the bias and add a 2nd Dimension
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    
    ## Sample the hidden nodes
    def sample_h(self, x):
        # Define product of the weights
        # .t = transpose which is used to make the equation mathematically correct
        wx = torch.mm(x, self.W.t())
        # expand_as = make the activation function the same Dimension for each mini-batch
        activation = wx + self.a.expand_as(wx)
        # Probability value given the visible nodes
        # Given the value of the visible nodes we return the probability of each of the hidden nodes = 1
        p_h_given_v = torch.sigmoid(activation)
        # Based on the probability, activate the hidden node
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    ## Sample the visible nodes
    def sample_v(self, y):
        # Define product of the weights
        wy = torch.mm(y, self.W)
        # expand_as = make the activation function the same Dimension for each mini-batch
        activation = wy + self.b.expand_as(wy)
        # Probability value given the hidden nodes
        # Given the value of the hidden nodes we return the probability of each of the visible nodes = 1
        p_v_given_h = torch.sigmoid(activation)
        # Based on the probability, predict whether the user will like the movie or not
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    ## Contrastive Divergence
    # v0 = Input vector, e.g. ratings of all the movies by one user
    # vk = Visible nodes after k sampling
    # ph0 = Vector of probabilities, at first iteration the hidden nodes = 1 given the values of v0
    # phk = Probabilities of the hidden nodes after k sampling
    def train(self, v0, vk, ph0, phk):
        # Updating the weights
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        # The sum is used to keep the same dimensions of the bias
        self.b += torch.sum( (v0 - vk), 0 )
        self.a += torch.sum( (ph0 - phk), 0 )

nv = len(training_set[0])
nh = 100
batch_size = 100

rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    ### Maximum Absolute Value
    train_loss = 0
    s = 0.
    ## Batch learning
    for id_user in range(0, nb_users - batch_size, batch_size):
        # Grabs the batch of units
        vk = training_set[id_user:id_user + batch_size]
        v0 = training_set[id_user:id_user + batch_size]
        # [variable name],_ = Only return first element of the function
        # Used to start the loop to make the Gibbs Chain for Gibbs Sampling
        ph0,_ = rbm.sample_h(v0)
        ## K-step Contrastive Divergence
        for k in range(10):
            # _,[variable name] = Only return second element of the function
            # returns sample of hidden nodes
            _,hk = rbm.sample_h(vk)
            # Returns sample of visible nodes
            _,vk = rbm.sample_v(hk)
            # Avoid using the ratings with -1 (Movies a user hasn't rated)
            vk[v0 < 0] = v0[v0 < 0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        ## Update the train loss
        # Identify the absolute value of the ratings that exist (that are not -1)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        # Update the counter to normalize the train loss
        s += 1.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss/s))
    
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1]
    # target - compare to predictions
    vt = test_set[id_user:id_user + 1]
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
print('Test loss: ' + str(test_loss/s))