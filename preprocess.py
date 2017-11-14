# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:15:52 2017

@author: user
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import keras.preprocessing.image

filename = 'processed\\train.json'
dir_train_ship = 'train\\ship\\'
dir_train_ice = 'train\\ice\\'
dir_test_ship = 'test\\ship\\'
dir_test_ice = 'test\\ice\\'

df = pd.read_json(filename,orient='records')
df = df.sample(frac=1).reset_index(drop=True) # shuffle

nData = len(df.index)
trainFrac = 0.8
nTrain = np.round(nData*trainFrac)

# generate images
x_train = []
y_train  = []
x_test = []
y_test = []

for index, row in df.iterrows():
    imag1 = row['band_1']
    imag2 = row['band_2']
    img1 = np.array(imag1)
    img2 = np.array(imag2)
    img1 = np.reshape(img1,(75,-1))
    img2 = np.reshape(img2,(75,-1))
    img = np.array([img1,img2,np.zeros(img1.shape)]).transpose()
    #img = np.array([img1,img2]).transpose()
    #img = np.array([img1,img1,img1]).transpose()
    if (index < nTrain):
        x_train.append(img)
        if (row['is_iceberg'] == 1):
            #mpimg.imsave(dir_train_ice + "ice" + str(index) +".png", img)
            y_train.append(1) 
        if (row['is_iceberg'] == 0):
            #mpimg.imsave(dir_train_ship + "ship" + str(index) +".png", img)    
            y_train.append(0) 
    else:
        x_test.append(img)
        if (row['is_iceberg'] == 1):
            #mpimg.imsave(dir_test_ice + "ice" + str(index) +".png", img)
            y_test.append(1) 
        if (row['is_iceberg'] == 0):
            #mpimg.imsave(dir_test_ship + "ship" + str(index) +".png", img)  
            y_test.append(0)             

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_train)
y_test = np.array(y_train)

import pickle
pickling_on = open("data_3d.pickle","wb")
pickle.dump([x_train,y_train,x_test,y_test], pickling_on)
pickling_on.close()


# =============================================================================
# with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([obj0, obj1, obj2], f)
# 
# # Getting back the objects:
# with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
#     obj0, obj1, obj2 = pickle.load(f)
# 
# 
# 
# =============================================================================




     