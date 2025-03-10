"""

"""
import numpy as np
import pandas as pandas
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import os
import math
import torch
import sklearn

# ---- Hyperparameters ----
n_estimators=100 
max_features="sqrt", 
max_depth=6, 
max_leaf_nodes=6

# ---- Loading Data ----

data = pandas.read_csv("archive/kaggle_3m/data.csv")
print(data.shape)
data.info()

X = data['Patient']
Y = data['death01']

# 80/20 split of data in train and test datasets
train_len = int(len(data) * 0.8)
train = data.iloc[:train_len, :]
test = data.iloc[train_len:, :]

# ---- Random Forest Bagging ----

# Maybe we can recursively build each tree when Node is inited?
# That could be one way hopefully it isn't computationally heavy 
# I think it's best to only build each tree in a random forest less than 10 layers deep so it shouldn't be that bad
# Yeah we can try that 
# We're doing a weird recursive solution that builds the tree when node is initalized
# I think we're training it to predict patient death given all the other info in the dataset
# We'll use the training data points for this
# I don't think we are now unless you know how to with random forest
# Cause I don't
# I don't think random forest is usually that good with images
# For that we should use ConvNets
# Yea true, maybe there's some way to create an image embedding and use that in a decision tree?
# I'm not sure how much data we have on tumor location in the dataset, I didn't see any
class RandomForest:
    def __init__(self, dataset, n_estimators=100):
        self.n_estimators = n_estimators
        self.trees = []
        self.features = []
        for i in range(n_estimators):
            boot = []
            for i in range(data.shape[0]):
                datapoint = np.random.randint(0, data.shape[0])
                boot.append(datapoint)
            # Not sure what to put for root here
            tree = Node(0, boot)
            self.trees.append(tree)
    
    def predict(self, dataset):
        predictions = []
        for tree in self.trees:
            pass


class Node:
    def __init__(self, depth, src, rows=[], cols=None, max_features="sqrt", left=None, right=None):
        self.depth = depth
        if src.shape[1] <= 1:
            self.left = None
            self.right = None
            return
        
        # Randomly selects a root from the available columns.
        self.root = np.random.randint(0, src.shape[1])

        # Partition the remaining nodes into subarrays.
        l_nodes = src.iloc[:, :self.root]
        r_nodes = src.iloc[:, self.root:]

        # Create left/right children from subsets. It is glorious.
        self.left = Node(depth+1, l_nodes)
        self.right = Node(depth+1, r_nodes)
                
root = Node(0, train)
print(root)
'''
Data columns (total 18 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Patient                    110 non-null    object 
 1   RNASeqCluster              92 non-null     float64
 2   MethylationCluster         109 non-null    float64
 3   miRNACluster               110 non-null    int64  
 4   CNCluster                  108 non-null    float64
 5   RPPACluster                98 non-null     float64
 6   OncosignCluster            105 non-null    float64
 7   COCCluster                 110 non-null    int64  
 8   histological_type          109 non-null    float64
 9   neoplasm_histologic_grade  109 non-null    float64
 10  tumor_tissue_site          109 non-null    float64
 11  laterality                 109 non-null    float64
 12  tumor_location             109 non-null    float64
 13  gender                     109 non-null    float64
 14  age_at_initial_pathologic  109 non-null    float64
 15  race                       108 non-null    float64
 16  ethnicity                  102 non-null    float64
 17  death01                    109 non-null    float64
'''


# Builds each tree
for i in range(n_estimators):
    break
    
# ---- Graphing Results ----
plt.plot(X, Y)
plt.show()