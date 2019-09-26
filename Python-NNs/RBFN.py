#!/usr/bin/env python
# coding: utf-8

# # This script trains an RBF Network on an example dataset

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd


# In[10]:


df = pd.read_csv('dataset.csv',header=None)
df.columns = ['f1','f2','res']
df.head()


# In[62]:


def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)


# In[63]:


def kmeans(X,k):
    clusters = np.random.choice(np.squeeze(X), size = k)
    prevClusters = clusters.copy()
    stds = np.zeros(k)
    converged = False
    
    while not converged:
        dist = np.squeeze(np.abs(X[:,np.newaxis]-clusters[np.newaxis,:]))
        closestCluster = np.argmin(dist, axis=1)
        for i in range(k):
            pointsCluster = X[closestCluster == i]
            if(len(pointsCluster)>0):
                clusters[i]=np.mean(pointsCluster,axis = 0)
        
        converged = np.linalg.norm(clusters - prevClusters)<0.000001
        prevClusters = clusters.copy()
        
    dist = np.squeeze(np.abs(X[:,np.newaxis]-clusters[np.newaxis,:]))
    closestCluster = np.argmin(dist,axis=1)
    emptyCluster = []
    for i in range(k):
        pointsCluster = X[closestCluster == i]
        if(len(pointsCluster)<2):
            emptyCluster.append(i)
            raise ClusterCountError('k is too big')
        else:
            stds[i]=np.std(pointsCluster, axis=0)
    return clusters, stds


# In[64]:


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)
    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k)
        else:
            # use a fixed std 
            self.centers, _ = kmeans(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
                print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()

                # update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)


# In[69]:


NUM_SAMPLES = 100
X = np.random.uniform(0., 1., NUM_SAMPLES)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = np.sin(2 * np.pi * X)  + noise
 
rbfnet = RBFNet(lr=1e-2, k=2)
rbfnet.fit(X, y)
 
y_pred = rbfnet.predict(X)
 
plt.plot(X, y, '-o', label='true')
plt.plot(X, y_pred, '-o', label='RBF-Net')
plt.legend()
 
plt.tight_layout()
plt.show()


# In[ ]:




