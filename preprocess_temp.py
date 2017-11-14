# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:15:52 2017

@author: user
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

filename = 'processed\\train.json'
df = pd.read_json(filename,orient='records')


imag1 = df.iloc[3]['band_1']
imag2 = df.iloc[3]['band_2']
img1 = np.array(imag1)
img2 = np.array(imag2)
img1 = np.reshape(img1,(75,-1))
img2 = np.reshape(img2,(75,-1))
img = np.array([img1,img2])

#img1 = pow(10,img1/20)
img = np.reshape(img,(75,-1))




imgplot = plt.imshow(img)
plt.show()
imgplot = plt.imshow(img2)
plt.show()

img
mpimg.imsave("out.png", img)
# check mean and variance of images
nData = len(df.index)
mean1 = []
mean2 = []
std1 = []
std2 = []
for index, row in df.iterrows():
    mean1.append(np.mean(row['band_1']))
    std1.append(np.std(row['band_1']))
    mean2.append(np.mean(row['band_2']))
    std2.append(np.std(row['band_2']))
    
     