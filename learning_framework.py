#!/usr/bin/python
from __future__ import print_function
from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution

import numpy as np
import cv2
import itertools as it

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json

# image parameters
downsampled_x = 64 #64
downsampled_y = 48#48
channels = 3 #channels on input image considered (GRAY8 = 1; RGB = 3)
skiprate = 3


class CustomDoomGame:
	def __init__(self,game,scenario, config,selectedMap):
		game.set_vizdoom_path("../ViZDoom/bin/vizdoom")
		game.set_doom_game_path("../ViZDoom/scenarios/doom2.wad")
		game.set_doom_scenario_path(scenario)
		game.set_doom_map(selectedMap)
		game.load_config(config)

def start_game(game,multiplayer,visible):
	if not visible:
		game.set_screen_resolution(ScreenResolution.RES_160X120)
	else:
		game.set_screen_resolution(ScreenResolution.RES_640X480)
	game.set_window_visible(visible)
	
	if multiplayer:
		# Start multiplayer game only with Your AI (with options that will be used in the competition, details in cig_host example).
		game.add_game_args("-host 1 -deathmatch +timelimit 10.0 "
			"+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
		# Name Your AI.
		game.add_game_args("+name AI")
		game.init()
	else:
		game.init()

# Function for converting images
def convert(img):
	'''
	#for GRAY8 images
	img = img[0].astype(np.float32) / 255.0
	img = cv2.resize(img, (downsampled_x, downsampled_y))
	return img
	'''
	# for RGB images
	img = img.astype(np.float32) / 255.0
	img_p = []
	for channel in range(channels):
		img_p.append(cv2.resize(img[:,:,channel], (downsampled_x, downsampled_y)))
	return np.array(img_p)


def create_cnn(input_rows,input_cols,num_outputs):
	model = Sequential()

	# input: input_colsxinput_rows images with 1 channels
	# this applies 32 convolution filters of size 3x3 each.
	'''
	model.add(Convolution2D(4, 5, 5,
                        border_mode='valid',
                        input_shape=(channels, input_rows, input_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(8, 4, 4))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(num_outputs))
	model.add(Activation('tanh'))
	model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
	'''
	model.add(Convolution2D(8, 3, 3,
                        border_mode='valid',
                        input_shape=(channels, input_rows, input_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) #32x24
	model.add(Convolution2D(8, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) #16x12
	model.add(Convolution2D(6, 2, 2))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) #8x6
	#model.add(Convolution2D(3, 2, 2))
	#model.add(Activation('tanh'))
	#model.add(MaxPooling2D(pool_size=(2, 2))) #4x3
	model.add(Flatten())
	model.add(Dense(num_outputs))
	model.add(Activation('tanh')) # num_outputs
	
	model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
	return model


def get_available_actions(num_actions):
	# Creates all possible actions.
	actions = []
	for perm in it.product([0, 1], repeat=num_actions):
		actions.append(list(perm))
	return actions

