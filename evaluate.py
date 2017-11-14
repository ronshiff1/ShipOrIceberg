# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:21:10 2017

@author: ronshiff
"""
import keras
import pandas as pd 
import numpy as np
import time

file_out = 'submission.csv'
# load model
model = keras.models.load_model('model1')

filename = 'test.json/data/processed/test.json'
df = pd.read_json(filename,orient='records')
df_csv = pd.read_csv('sample_submission.csv')


#
start = time.time()

vec = []
#
for index, row in df_csv.iterrows():
    item2eval = df[df['id'] == row['id']].iloc[0]
    imag1 = item2eval['band_1']
    imag2 = item2eval['band_2']
    img1 = np.array(imag1)
    img2 = np.array(imag2)
    img1 = np.reshape(img1,(75,-1))
    img2 = np.reshape(img2,(75,-1))
    img = np.array([img1,img2,np.zeros(img1.shape)]).transpose()
    img = (img - np.mean(img))/np.std(img)
    vec.append(img)
   
#x = np.expand_dims(img, axis=0) #add dimension to be 4 dim
xx = np.array(vec)
prob = model.predict_proba(xx)
#
for index, row in df_csv.iterrows():
    df_csv.set_value(index,'is_iceberg',prob[index]) 

df_csv.to_csv(file_out, sep=',', index = False ,encoding='utf-8')    
    
    
# =============================================================================
#     
# end = time.time()
# print(end - start)   
# 
#     
# cnt = 0
# 
# #
# import time
# start = time.time()
# 
# #
# for index, row in df_csv.iterrows():
#     cnt += 1
#     print(cnt)
#     if (cnt == 50):
#         break
#     item2eval = df[df['id'] == row['id']].iloc[0]
#     imag1 = item2eval['band_1']
#     imag2 = item2eval['band_2']
#     img1 = np.array(imag1)
#     img2 = np.array(imag2)
#     img1 = np.reshape(img1,(75,-1))
#     img2 = np.reshape(img2,(75,-1))
#     img = np.array([img1,img2,np.zeros(img1.shape)]).transpose()
#     img = (img - np.mean(img))/np.std(img)
#     x = np.expand_dims(img, axis=0) #add dimension to be 4 dim
#     prob = model.predict_proba(x)
#     df_csv.iloc[index]['is_iceberg'] = prob
#     
# end = time.time()
# print(end - start)
#     
# =============================================================================
