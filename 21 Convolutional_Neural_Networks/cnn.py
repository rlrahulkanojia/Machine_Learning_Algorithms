# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:03:58 2018

@author: Abstergo
"""

# Part 1- Building the convolutional network

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

'''Initialsing the CNN'''
classifier = Sequential()

''' Step 1 - Convulutional'''
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3) , activation = 'relu'))

#input_shape - An important parameter as it specifies the expected format of the image by CNN
#the first param  represents the number of layers 3 for color image (RGB channel) 
# and 2 for black and white. The next two param represnts the dimensions of each channel

#The order of param is different in theano and tensorflow backend
# In theano it is as mentioned above but in Tensorflow dimensions comes first and then number of channels

#In ANN we used activation function to activate the neuron in nural network but in this we use activation function
# to make sure that we dont have any negative pixel values on feature map.
# We have to remove these negative pixel in order have non-linearity in CNN.

# Step - 2 Pooling - Reducing the size of feature map
# We reduce it by again applying feature detector on feature map(Convolutional Operation)

#Input Image --Convolution-> Convolutional Layer --Pooling--> Pooling Layer --Flattening--> Single Input Layer Of ANN -> Fully COnnected Layer(Hidden Layer )-> Output Layer

classifier.add(MaxPooling2D(pool_size = (2, 2)))#Most Time 2X2 dimesin is taken as pool_size

#'''Step 3- Flattening'''
classifier.add(Flatten())

#'''Step - 4 -> Fully COnnected layer(Hidden Layer) '''
classifier.add(Dense(output_dim = 128 ,activation = 'relu'))#Output_dim = Expermenting show 100 around and power of 2 is better choice
classifier.add(Dense(output_dim = 2 ,activation = 'sigmoid'))

#Compliling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part-2 Fitting the CNN to the Images
#We will use keras lib as shortcut to prevent over fitting
# Image Augumentaion creates batches o find a pattern which overcomes the 
# need of large dataset of images(appox 100000-500000 images) to 10000 
#`Images are processed in various ways sucj as rotation, shearing c=increasing the dataset availavle for training
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)