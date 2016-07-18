#!/usr/bin/python
from __future__ import print_function
from vizdoom import *

from random import choice,sample
import random
from time import sleep
from time import time
import time

import numpy as np
from six.moves import cPickle
import cv2

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json, Model
from keras.callbacks import EarlyStopping

import MultiNEAT as NEAT
from MultiNEAT import GetGenomeList, ZipFitness, EvaluateGenomeList_Serial, EvaluateGenomeList_Parallel

from learning_framework import *
import learning_framework

image_data_filename = "feature_images/cig_rgb.dat"
autoencoder_weights_filename = "autoencoders/encoder_weights_8x8x6.save"
autoencoder_topology_filename = "autoencoders/topologies/encoder_topology_8x8x6.save"

def train_autoencoder():
	#http://blog.keras.io/building-autoencoders-in-keras.html
	input_img = Input(shape=(channels, downsampled_y, downsampled_x))

	x = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x) 
	x = MaxPooling2D((2, 2), border_mode='same')(x) 
	x = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)

	# at this point the representation is (8, 8, 6) i.e. 384-dimensional

	x = Convolution2D(4, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x) 
	x = Convolution2D(24, 3, 3, activation='relu')(x) 
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(channels, 3, 3, activation='sigmoid', border_mode='same')(x)
	
	autoencoder = Model(input=input_img, output=decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

	f = open(image_data_filename,'rb')
	training_img_set = cPickle.load(f)
	f.close()
	
	x_train = np.array(training_img_set).reshape([len(training_img_set),channels,downsampled_y,downsampled_x])
	np.random.shuffle(x_train)
	
	early_stopping = EarlyStopping(monitor='val_loss', patience=50000,mode='min')
	hist = autoencoder.fit(x_train,x_train,nb_epoch=3000,batch_size=4,shuffle=True,verbose=1) #validation_split=0.01,callbacks=[early_stopping]
	
	#print(hist.history['loss'])
	
	#test
	decoded_imgs = autoencoder.predict(x_train)
	#compare x_train vs decoded_imgs
	for i in range(len(decoded_imgs)):
		original = cv2.resize(x_train[i].reshape(downsampled_y,downsampled_x,channels), (320,200), interpolation = cv2.INTER_AREA)
		cv2.imshow('Doom Buffer',original)
		cv2.waitKey(500)
		decoded = cv2.resize(decoded_imgs[i].reshape(downsampled_y,downsampled_x,channels), (320,200), interpolation = cv2.INTER_AREA)
		cv2.imshow('Doom Buffer',decoded)
		cv2.waitKey(500)
	cv2.destroyAllWindows()
	
	#saving model and weights
	encoder = Model(input=input_img,output=encoded)
	encoder.save_weights(autoencoder_filename,overwrite=True)
	json_string = encoder.to_json()
	f = open(autoencoder_topology_filename,'wb')
	cPickle.dump(json_string,f,protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
	
	for layer in encoder.layers:
		print("*******",layer.output)


train_autoencoder()



