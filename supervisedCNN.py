#!/usr/bin/python
#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
from __future__ import print_function
from vizdoom import *
import cv2

from random import choice
from time import sleep
from time import time

import numpy as np
from six.moves import cPickle
import h5py

from learning_framework import *
import learning_framework

doom_scenario = "scenarios/cig_orig_pistol.wad"
playable_doom_config = "config/cig_playable.cfg"
non_playable_doom_config = "config/cig.cfg"
mapSelected = "map01"
training_data_filename = 'supervised_CNN/cig_orig_pistol_marine_training_data.save'
cnn_layout_filename = 'supervised_CNN/cig_orig_pistol_marine_cnn_layout.net'
cnn_weights_filename = 'supervised_CNN/cig_orig_pistol_marine_cnn_weights.h5'

isCig = True

# what task to do
gather_data = False 
trainOrLoad = False #True for training, False for loading pretrained network

training_batch = 64
training_epochs = 1000

episodes_recorded = 20																							
number_actions = 4

training_img_set = []
labels_set = []

def gatherData(game):
	for i in range(episodes_recorded):
		print("Episode #", i+1)

		# Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
		game.new_episode()

		while not game.is_episode_finished():

			# Gets the state
			s = game.get_state()

			# Makes a random action and get remember reward.
			game.advance_action(1)
			action = game.get_last_action()
			if s.number % 2 == 0:
				continue
			if all(v==0 for v in action[:number_actions]):
				#skip frame
				continue

			print("action:", action[:number_actions])

			# Get processed image
			# Gray8 shape is not cv2 compliant
			#img = convert(s.image_buffer)
			img = convert(s.image_buffer,False)
			#rows = len(img[0])
			#cols = len(img[0][0])

			# store image and action
			training_img_set.append(img)
			labels_set.append(action)
		print("Episode finished: ",i+1)

	# End data gathering
	# store data in format: http://deeplearning.net/tutorial/gettingstarted.html
	# create training data as numpy arrays of input and labels
	print("Data gathering finished.")
	print("Total training datapoints:", len(training_img_set))
	print("Total labels:", len(labels_set))
	print("************************")

	# pickle (record) data
	f = open(training_data_filename,'wb')
	cPickle.dump(training_img_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(labels_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()


def generate_network():
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Convolution2D, MaxPooling2D
	from keras.optimizers import SGD

	model = Sequential()
	
	# input: 144x256 images with 1 channels -> (1, 100, 100) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Convolution2D(24, 5, 5,
                        border_mode='valid',
                        input_shape=(channels, downsampled_y, downsampled_x)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 4, 4))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	#model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation('relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(number_actions))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

	# to store and load network layout:
	json_string = model.to_json()
	f = open(cnn_layout_filename,'wb')
	cPickle.dump(json_string,f,protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
	return model

def load_network_layout(filename):
	from keras.models import model_from_json
	f = open(filename,'rb')
	json_string = cPickle.load(f)
	f.close()
	model = model_from_json(json_string)
	return model

def load_network_weights(filename,net):
	net.load_weights(filename)

def train_network(net):
	#load training data
	f=open(training_data_filename,'rb')
	tr=cPickle.load(f)
	lb=cPickle.load(f)
	f.close()
	print("Loaded training datapoints:", len(tr))
	print("Loaded labels:", len(lb))
	tr = np.array(tr).reshape([len(tr), channels, downsampled_y, downsampled_x])
	lb = np.array(lb)[:,0:number_actions] ########################### TEMPORARY?
	net.fit(tr, np.array(lb), batch_size=training_batch, nb_epoch=training_epochs, verbose=1)
	loss_and_metrics = net.evaluate(np.array(tr), np.array(lb), batch_size=32)
	net.save_weights(cnn_weights_filename,overwrite=True)
	return loss_and_metrics


# Create DoomGame instance
game = DoomGame()
CustomDoomGame(game,doom_scenario,playable_doom_config,mapSelected)

if gather_data:
	start_game(game,isCig,True,Mode.SPECTATOR)
	gatherData(game)

if trainOrLoad:
	cnn_network = generate_network()
	loss_and_metrics = train_network(cnn_network)
else:
	cnn_network = load_network_layout(cnn_layout_filename)
	load_network_weights(cnn_weights_filename,cnn_network)

game.close()

start_game(game,isCig,True)

test_episodes = 5
sleep_time = 0.028

for i in range(test_episodes):
    print("Episode #" + str(i+1))
    game.new_episode()

    while not game.is_episode_finished():

        s = game.get_state()

        # Get processed image
        img = convert(s.image_buffer,False).reshape([1, channels, downsampled_y, downsampled_x])
        network_input = []
        network_input.append(img)
        output = cnn_network.predict(img)#np.array(network_input))
        output[output>0.4] = 1
        output[output<=0.4] = 0
        output = output.flatten()
        print("Action selected:",output)
        action = [0 for _ in range(number_actions)]
        for i in range(number_actions):
            action[i] = int(output[i])
        r = game.make_action(action)
        sleep(sleep_time)
        # Display the image here!
        #if game.get_screen_format() in [ScreenFormat.GRAY8, ScreenFormat.DEPTH_BUFFER8]:
        #    img = img.reshape(img.shape[1],img.shape[2],1)

        #cv2.imshow('Doom Buffer',img)
        #cv2.waitKey(int(sleep_time * 1000))
        
    # Check how the episode went.
    print("Episode finished.")
    print("total reward:", game.get_total_reward())
    print("************************")

cv2.destroyAllWindows()
game.close()




#for i in range(episodes):
    #print("Episode #" + str(i+1))

    ## Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    #game.new_episode()

    #while not game.is_episode_finished():

        ## Gets the state
        #s = game.get_state()

        ## Makes a random action and get remember reward.
        #r = game.make_action(choice(actions))

        ## Prints state's game variables. Printing the image is quite pointless.
        #print("State #" + str(s.number))
        #print("Game variables:", s.game_variables[0])
        #print("Reward:", r)
        #print("=====================")
        
        ## Get processed image
        ## Gray8 shape is not cv2 compliant
        #img = s.image_buffer
        #rows = len(img[0])
        #cols = len(img[0][0])
        #print("resolution value: " + str(img[0][rows/2][cols/2]))
        
        ## Display the image here!
        #if game.get_screen_format() in [ScreenFormat.GRAY8, ScreenFormat.DEPTH_BUFFER8]:
            #img = img.reshape(img.shape[1],img.shape[2],1)

        #cv2.imshow('Doom Buffer',img)
        #cv2.waitKey(int(sleep_time * 1000))
        
        ##if sleep_time>0:
        ##    sleep(sleep_time)

    ## Check how the episode went.
    #print("Episode finished.")
    #print("total reward:", game.get_total_reward())
    #print("************************")

#cv2.destroyAllWindows()

# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
#game.close()
