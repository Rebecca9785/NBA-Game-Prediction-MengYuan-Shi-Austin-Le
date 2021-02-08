#importing libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from src.layers.layers import *


#implement graph convolutional network
class Graph1Net(nn.Module):
    def __init__(self, in_features, num_hidden, out_features, bias = False):
        super(Graph1Net, self).__init__()
        self.gc1 = GraphConvolution(in_features, num_hidden, bias)
        self.gc2 = GraphConvolution(num_hidden, out_features, bias)

    def forward(self, x, adj_hat):
        x = F.relu(self.gc1(x, adj_hat))
        return self.gc2(x, adj_hat)

#implement graph convolutional network with a hidden layer
class Graph2Net(nn.Module):
    def __init__(self, in_features, num_hidden, out_features, bias = False):
        super(Graph2Net, self).__init__()
        self.gc1 = GraphConvolution(in_features, num_hidden, bias)
        self.gc2 = GraphConvolution(num_hidden, num_hidden, bias)
        self.gc3 = GraphConvolution(num_hidden, out_features, bias)

    def forward(self, x, adj_hat):
        x = F.relu(self.gc1(x, adj_hat))
        x = F.relu(self.gc2(x, adj_hat))
        return self.gc3(x, adj_hat)
    
#implement GarphSAGE
class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())
