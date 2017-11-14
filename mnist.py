# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:17:30 2017

@author: user
"""

from keras.models import Sequential                                        
from keras.layers.core import Dense, Activation                                                              
from keras.utils import np_utils
from sklearn import datasets

iris = datasets.load_iris()    
print iris.data.shape
print iris.target.shape                                                       

model = Sequential()                                                       
model.add(Dense(4, 3, init='uniform'))                                   
model.add(Activation('softmax'))                                           

model.compile(loss='mean_squared_error', optimizer='rmsprop')                    

labels = np_utils.to_categorical(iris.target)                                              
model.fit(iris.data, labels, nb_epoch=5, batch_size=1, show_accuracy=True, validation_split=0.3)