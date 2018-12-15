#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 21:41:41 2018

@author: bolof
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

# create your independent and dependent variables of X and y

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#no need for splitting the data to train and test sets since we have very few datasets

#feature scaling 



#fittting the linear regression to the dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)

