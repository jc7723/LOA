import pandas as pd
import numpy as np
#import math
#import matplotlib
#import matplotlib.pyplot as plt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#from sklearn import datasets, linear_model, preprocessing, model_selection
#import numpy.polynomial.polynomial as poly
#import scipy.io
#import sklearn
#import random
#import pickle



def iso_generator():
    #x = pd.read_csv('isoletx 1.csv', header=None).values
    #y = pd.read_csv('isolety.csv', header=None).values
    x = pd.read_csv('aloix.csv', header=None).values
    y = pd.read_csv('aloiy.csv', header=None).values
    data_size = len(y)
    shuffle = np.random.permutation(data_size)
    x = x[shuffle]
    y = y[shuffle]
    del shuffle
    #normalize
    #x = x-np.reshape(np.mean(x, axis=1), (data_size, 1))
    x = x/np.expand_dims(np.std(x, axis=1), axis=1)
    x[np.isinf(x)] = 0
    x = x - np.mean(x)
    for i in range(data_size):
        yield (x[i], y[i])
    

def mnist_generator():
    
    #load mnist
    from sklearn.datasets import fetch_mldata
    dataset = fetch_mldata("MNIST original")
    data_size = dataset.target.shape[0]
    #permutation
    shuffle = np.random.permutation(data_size)
    data = dataset.data[shuffle]
    label = dataset.target[shuffle]
    del shuffle, dataset
    
    #normalize
    #data = data-np.reshape(np.mean(data, axis=1), (data_size, 1))
    data = data/np.reshape(np.std(data, axis=1), (data_size, 1))
    data = data-np.mean(data)
    '''
    #resize 784 to 28*28
    data2 = np.zeros((data_size, 28, 28))
    for i in range (data_size):
    data2[i, ] = np.reshape(data[i, ], (28, 28))
    data = data2
    del data2
    '''
    for i in range(data_size):
        yield (data[i], label[i])


class fc_with_sigmoid(object):
    
    def __init__(self):
        self.w = .000001*(np.random.rand(128)-.5)
        self.b0 = .000001*(np.random.random()-.5)
        
    def test(self, x):
        y_hat = self.w.dot(x) + self.b0
        #print('y_hat =', y_hat)
        return 1/(1+np.exp(-y_hat))
    
    def train(self, x, c, lr=.05):
        y_hat = self.w.dot(x) + self.b0
        c_hat = 1/(1+np.exp(-y_hat))
        #print('w_before =', self.w[:5])
        temp = lr*(c_hat - c)/(np.exp(-y_hat)+1)/(np.exp(y_hat)+1)
        self.w = self.w - temp*x
        #print('w_after =', self.w[:5])
        #print('b0_before =', self.b0)
        self.b0 = self.b0 -temp
        #print('b0_after =', self.b0)



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
        self.ex = []
        self.c_his = []
        self.suspend_n = []
        self.suspend_e = []
        self.suspend_class_name = []
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
        if class_name not in self.suspend_class_name:
            self.n.append(0)
            self.e.append(0)
            self.class_name.append(class_name)
            self.ex.append(0)
            self.c_his.append(int(0))
        else:
            index_in_suspend_class_name = self.suspend_class_name.index(class_name)
            self.n.append(self.suspend_n.pop(index_in_suspend_class_name))
            self.e.append(self.suspend_e.pop(index_in_suspend_class_name))
            self.class_name.append(self.suspend_class_name.pop(index_in_suspend_class_name))
            self.ex.append(0)
            self.c_his.append(int(0))
        
    def suspendClassInNode(self, class_name):
        class_index_in_node = self.class_name.index(class_name)
        self.suspend_n.append(self.n.pop(class_index_in_node))
        self.suspend_e.append(self.e.pop(class_index_in_node))
        self.suspend_class_name.append(self.class_name.pop(class_index_in_node))
        del self.ex[class_index_in_node]
        del self.c_his[class_index_in_node]
        
    def updateExpectation(self, c, class_index_in_node):
        self.n_all = self.n_all + 1
        self.e_all = self.e_all + c
        self.n[class_index_in_node] = self.n[class_index_in_node] + 1
        self.e[class_index_in_node] = self.e[class_index_in_node] + c
        self.ex[class_index_in_node] = self.e[class_index_in_node]/self.n[class_index_in_node]
        
    def findExceptionAll(self):
        if self.n_all == 0:
            return 0
        else:
            #mean
            #return self.e_all/self.n_all
            #medium
            #return np.median(np.array([self.findExceptionOneClass(i) for i in range(len(self.class_name))]))
            return np.median(self.ex)
        
    def findExceptionOneClass(self, class_index_in_node):
        if self.n[class_index_in_node] == 0:
            return 0
        else:
            return self.e[class_index_in_node]/self.n[class_index_in_node]
    
    def judgeInTrain(self, class_index_in_node):
        #c == 0: left, c == 1: right        
        return int(self.findExceptionAll() <= self.findExceptionOneClass(class_index_in_node))
        
        #just assign the classes, to make sure tree is balanced
        #return class_index_in_node%2
    
    def judgeInTest(self, x):
        return int(self.findExceptionAll() <= self.testModel(x))
    
    
    
    
        
class tree(object):
    
    def __init__(self):
        self.nodes = []
        self.current_node_index = 0
        self.root = node(self.current_node_index)
        self.nodes.append(self.root)
        self.current_node_index = self.current_node_index + 1
        
    def giveBirth(self, node_index):
        #node_index is index of node in self.nodes, who is going to give birth to left and right
        #two new born nodes would be appended at the end of self.nodes
        self.nodes.append(node(self.current_node_index))
        self.current_node_index = self.current_node_index + 1
        self.nodes[node_index].setLeft(self.nodes[-1])
        self.nodes[-1].setParent(self.nodes[node_index])
        
        self.nodes.append(node(self.current_node_index))
        self.current_node_index = self.current_node_index + 1
        self.nodes[node_index].setRight(self.nodes[-1])
        self.nodes[-1].setParent(self.nodes[node_index])
        
    def deleteChilds(self, node):
        delete_index = node.left.node_index
        del self.nodes[delete_index]
        self.current_node_index = self.current_node_index - 1
        for i in range(delete_index, self.current_node_index):
            self.nodes[i].node_index = i
        
        delete_index = node.right.node_index
        del self.nodes[delete_index]
        self.current_node_index = self.current_node_index - 1
        for i in range(delete_index, self.current_node_index):
            self.nodes[i].node_index = i
        
        node.left = None
        node.right = None
    
    def suspendClass(self, class_name, node):
        node.suspendClassInNode(class_name)
        if len(node.class_name) == 1:
            #if this node loses(suspends) its last 2nd class, delete both its childs
            self.deleteChilds(node)
        elif len(node.class_name) == 0:
            #attention that, if this node loses its last class, it will NOT be deleted, and nothing special will be done
            #this will lead to a special temporary status: this node includes no class, its parent and sibling have the same classes
            #this temporary status will not last long, unless expection of samples arrives at its parent keep increase monotonically, which is unlikely for a pre-normalized dataset
            #this temporary status may cause problem for testing, because a leaf of the decision tree might cannot even give out a single class, and it is fixed in onlineTest function
            pass
        else:
            #recursive down
            if class_name in node.left.class_name:
                self.suspendClass(class_name, node.left)
            elif class_name in node.right.class_name:
                self.suspendClass(class_name, node.right)
            else:
                print('suspendClass fail in node: ', node)

    def onlineTrain(self, xy, node):
        x, y = xy
        #print('node_index =', node.node_index)
        
        #1:register if y is new in this node
        if_register = (y not in node.class_name)
        if if_register:
            node.addClass(y)
            
        #2:judge (if current h(x) is above average)
        class_index_in_node = node.class_name.index(y)
        #print('class_index_in_node =', class_index_in_node)
        #c == 0: left, c == 1: right
        c = node.judgeInTrain(class_index_in_node)
        #print('c =', c)
        
        #3:train
        node.trainModel(x, c)
        
        #4:update e, n
        c_hat = node.testModel(x)
        node.updateExpectation(c_hat, class_index_in_node)
        
        #5:give birth if second class arrives at this node
        if if_register and len(node.class_name) == 2:
            self.giveBirth(node.node_index)
            #the only previous class should be arranged to other child
            #though it cannot be trained in an online algorithm
            [node.left, node.right][1-c].addClass(node.class_name[0])
            
        #6:suspend a class recursively if this class no longer go to a child of this node
        if not if_register and node.c_his[class_index_in_node] == 1-c and node.left != None and y in [node.left, node.right][1-c].class_name:#this class's c changes
            self.suspendClass(y, [node.left, node.right][1-c])
        node.c_his[class_index_in_node] = c

        #7:a sample trains recursively down all the way to a leaf
        del x, y, if_register, class_index_in_node, c_hat
        if node.left != None:#if a node have child, then it must have exactly two childs
            self.onlineTrain(xy, [node.left, node.right][c])

    def startOnlineTrain(self, xy):
        self.onlineTrain(xy, self.root)

    def onlineTest(self, x, node):
        if len(node.class_name) == 1:
            return node.class_name[0]
        else:
            try:
                return self.onlineTest(x, [node.left, node.right][node.judgeInTest(x)])
                return self.onlineTest(x, [node.left, node.right][1-node.judgeInTest(x)])
                print('onlineTest return fail at node: ', node)
            except:
                pass
        
    def startOnlineTest(self, x):
        return self.onlineTest(x, self.root)






if __name__ == "__main__":
    
    #build
    #my_generator = mnist_generator()
    my_generator = iso_generator()
    my_tree = tree()
    
    #train
    for i in range(90000):
        if i%500 == 0:
            print('index of sample i =', i)
        my_tree.startOnlineTrain(next(my_generator))
    
    #test
    test_result = []
    for i in range(10000):
        x, y = next(my_generator)
        test_result.append(int(y == my_tree.startOnlineTest(x)))
    accurancy = np.mean(test_result)
    print('accurancy =', accurancy)








