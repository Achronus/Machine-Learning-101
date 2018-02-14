# A3C AI - Breakout

# Importing the libraries 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0): # std stands for standard deviation
    out = torch.randn(weights.size()) # Initialize the weights
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) # Gets the weights, squares them and provides the normalized weights value using the std - var(out) = std^2
    return out

# Initalizing the weights of the neural network for optimal learning
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / fan_in + fan_out) # Represents the tensor of weights
        m.weight.data.uniform_(-w_bound, w_bound) # Create random weights
        m.bias.data.fill_(0) # Fills the bias with 0s
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out) # Represents the tensor of weights
        m.weight.data.uniform_(-w_bound, w_bound) # Create random weights
        m.bias.data.fill_(0) # Fills the bias with 0s
        
# Making the A3C brain
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1) # Output = V(s)
        self.actor_linear = nn.Linear(256, num_outputs) # Output = Q(s,a)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train() # Puts into train mode
    
    def forward(self, inputs):
        inputs, (hx, cx) = inputs # hx = Hidden states; cx = Cell states
        x = F.elu(self.conv1(inputs)) # Propagates images to first layer; elu = Exponential Linear Unit
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 3 * 3) # Flatten vector
        (hx, cx) = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)