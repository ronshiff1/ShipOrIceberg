# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:37:32 2017

@author: user
"""

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (75, 75, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units= 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range = 3, zoom_range = 0.2,samplewise_center=True,samplewise_std_normalization = True)


test_datagen = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization = True)

# =============================================================================
# training_set = train_datagen.flow_from_directory('train',
#                                                  target_size = (75, 75),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')
# # =============================================================================
import pickle
pickle_off = open("data_3d.pickle","rb")
[x_train,y_train,x_test,y_test] = pickle.load(pickle_off)
pickle_off.close()

training_set = train_datagen.flow(x = x_train, y = y_train , batch_size=32)

# =============================================================================
# test_set = test_datagen.flow_from_directory('test',
#                                             target_size = (75, 75),
#                                             batch_size = 32,
#                                             class_mode = 'binary')
# 
# =============================================================================
test_set = test_datagen.flow(x = x_test, y = y_test , batch_size=32)

classifier.fit_generator(training_set,
                         samples_per_epoch = 1000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples =500)


