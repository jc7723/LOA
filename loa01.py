import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets, linear_model, preprocessing, model_selection
import numpy.polynomial.polynomial as poly
import scipy.io
import sklearn
import random
import pickle







class fc_with_sigmoid(object):
    
    def __init__(self):
        self.w = np.random.rand(28**2)-.5
        self.b0 = np.random.random()-.5
        
    def test(self, x):
        y_hat = self.w.dot(x) + self.b0
        return 1/(1+np.exp(-y_hat))
    
    def train(self, x, c, lr=0.1):
        y_hat = self.w.dot(x) + self.b0
        c_hat = 1/(1+np.exp(-y_hat))
        self.w = self.w - lr*(c_hat - c)*(np.exp(-y_hat)/(np.exp(-y_hat+1))**2)*x
        self.b0 = self.b0 -lr*(c_hat - c)*(np.exp(-y_hat)/(np.exp(-y_hat+1))**2)


class node(object):
    
    def __init__(self, node_index):
        self.left = None
        self.right = None
        self.parent = None
        self.n_all = 0
        self.e_all = 0
        self.n = []
        self.e = []
        self.class_name = []
        self.node_index = node_index
        self.model = fc_with_sigmoid()
        
    def setLeft(self, node):
        self.left = node

    def setRight(self, node):
        self.right = node
        
    def setParent(self, node):
        self.parent = node
        
    def testModel(self, x):
        return self.model.test(x)
    
    def trainModel(self, x, c):
        self.model.train(x, c)
        
    def addClass(self, class_name):
        self.n.append(0)
        self.e.append(0)
        self.class_name.append(class_name)
        
    def updateExpectation(self, c, class_index_in_node):
        self.n_all = self.n_all + 1
        self.e_all = self.e_all + c
        self.n[class_index_in_node] = self.n[class_index_in_node] + 1
        self.e[class_index_in_node] = self.e[class_index_in_node] + c
        
        
        
class tree(object):
    
    def __init__(self):
        self.nodes = []        
        self.current_node_index = 0
        self.root = node(self.current_node_index)
        self.nodes.append(self.root)
        self.current_node_index = self.current_node_index + 1
        
    def giveBirth(self, node_index):
        self.nodes.append(node(self.current_node_index))
        self.current_node_index = self.current_node_index + 1
        self.nodes[node_index].setLeft(self.nodes[-1])
        self.nodes[-1].setParent(self.nodes[node_index])
        self.nodes.append(node(self.current_node_index))
        self.current_node_index = self.current_node_index + 1
        self.nodes[node_index].setRight(self.nodes[-1])
        self.nodes[-1].setParent(self.nodes[node_index])

    def onlineTrain(self, x, y, node):
        #1:register if y is new in this node
        if_register = (y not in node.class_name)
        if if_register:
            node.addClass(y)
        #2:judge (if current h(x) is above average)
        class_index_in_node = node.class_name.index(y)
        if node.n_all == 0:
            c = int(0 <= 0)
        elif node.n[class_index_in_node] == 0:
            c = int(node.e_all/node.n_all <= 0)
        else:
            c = int(node.e_all/node.n_all <= node.e[class_index_in_node]/node.n[class_index_in_node])
        #3:train
        node.trainModel(x, c)
        #4:update e, n
        c_hat = node.testModel(x)
        node.updateExpectation(c_hat, class_index_in_node)
        #5:give birth if second class arrives at this node
        if if_register and len(node.class_name) == 2:
            self.giveBirth(node.node_index)
            [node.left, node.right][1-c].class_name.append(node.class_name[0])
        #6:recursive
        if node.left != None:
            self.onlineTrain(x, y, [node.left, node.right][c])

    def startOnlineTrain(self, x, y):
        self.onlineTrain(x, y, self.root)





if __name__ == "__main__":
    
    #load mnist
    from sklearn.datasets import fetch_mldata
    dataset = fetch_mldata("MNIST original")
    data_size = dataset.target.shape[0]
    #permutation
    shuffle = np.random.permutation(data_size)
    data = dataset.data[shuffle]
    label = dataset.target[shuffle]
    del shuffle
    '''
    #resize 784 to 28*28
    data2 = np.zeros((data_size, 28, 28))
    for i in range (data_size):
        data2[i, ] = np.reshape(data[i, ], (28, 28))
    data = data2
    del data2
    '''

    
    my_tree = tree()
    for i in range(70000):
        my_tree.startOnlineTrain(data[i], label[i])












