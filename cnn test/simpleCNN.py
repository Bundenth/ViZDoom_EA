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

downsampled_x = 256
downsampled_y = int(9/16.0*downsampled_x)


# Function for converting images
def convert(img):
    img = img[0].astype(np.float32) / 255.0
    img = cv2.resize(img, (downsampled_x, downsampled_y))
    return img


training_img_set = []
labels_set = []

def gatherData(game,episodes_recorded,filename):
	for i in range(episodes_recorded):
		print("Episode #", i+1)

		# Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
		game.new_episode()

		while not game.is_episode_finished():

			# Gets the state
			s = game.get_state()

			# Makes a random action and get remember reward.
			game.advance_action()
			action = game.get_last_action()

			if all(v==0 for v in action[:3]):
				#skip frame
				continue

			print("action:", action[:3])

			# Get processed image
			# Gray8 shape is not cv2 compliant
			#img = convert(s.image_buffer)
			img = s.image_buffer / 255.0
			#rows = len(img[0])
			#cols = len(img[0][0])

			# store image and action
			training_img_set.append(img)
			labels_set.append(action[:3])
		print("Episode finished: ",i+1)

	# End data gathering
	# store data in format: http://deeplearning.net/tutorial/gettingstarted.html
	# create training data as numpy arrays of input and labels
	print("Data gathering finished.")
	print("Total training datapoints:", len(training_img_set))
	print("Total labels:", len(labels_set))
	print("************************")

	# pickle (record) data
	f = open(filename,'wb')
	cPickle.dump(training_img_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
	cPickle.dump(labels_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()


def generate_network(filename):
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation, Flatten
	from keras.layers import Convolution2D, MaxPooling2D
	from keras.optimizers import SGD

	model = Sequential()
	
	# input: 144x256 images with 1 channels -> (1, 100, 100) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Convolution2D(32, 3, 3,
                        border_mode='valid',
                        input_shape=(1, 200, 320)))
	model.add(Activation('relu'))
	model.add(Convolution2D(48, 5, 5))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(3))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

	# to store and load network layout:
	json_string = model.to_json()
	f = open(filename,'wb')
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

def train_network(training_set_filename,net,weights_filename_dst):
	#load training data
	f=open(training_set_filename,'rb')
	tr=cPickle.load(f)
	lb=cPickle.load(f)
	f.close()
	print("Loaded training datapoints:", len(tr))
	print("Loaded labels:", len(lb))
	net.fit(np.array(tr), np.array(lb), batch_size=32, nb_epoch=18)
	loss_and_metrics = net.evaluate(np.array(tr), np.array(lb), batch_size=32)
	net.save_weights(weights_filename_dst,overwrite=True)
	return loss_and_metrics


# Create DoomGame instance. It will run the game and communicate with you.
#game = DoomGame()

## Now it's time for configuration!
#game.load_config("../../examples/config/cnnLearning.cfg")

## Adds buttons that will be allowed. 
#if game.get_mode() == Mode.PLAYER:
	#game.add_available_button(Button.MOVE_LEFT)
	#game.add_available_button(Button.MOVE_RIGHT)
	#game.add_available_button(Button.ATTACK)
#elif game.get_mode() == Mode.SPECTATOR:
	#game.add_available_button(Button.TURN_LEFT)
	#game.add_available_button(Button.TURN_RIGHT)
	#game.add_available_button(Button.ATTACK)
	#game.add_available_button(Button.STRAFE)
	#game.add_available_button(Button.MOVE_LEFT)
	#game.add_available_button(Button.MOVE_RIGHT)

## Initialize the game. Further configuration won't take any effect from now on.
#game.init()

#gatherData(game,100,'training_data.save')
#game.close()

#cnn_network = generate_network('cnn_test.cnn')
cnn_network = load_network_layout('cnn_test.cnn')
		
#loss_and_metrics = train_network('training_data.save',cnn_network,'network_weights.h5')
load_network_weights('network_weights.h5',cnn_network)

game = DoomGame()
game.load_config("../../../../examples/config/cnnLearning.cfg")
game.set_mode(Mode.PLAYER)
game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)
game.init()

episodes = 5
sleep_time = 0.028

for i in range(episodes):
    print("Episode #" + str(i+1))
    game.new_episode()

    while not game.is_episode_finished():

        s = game.get_state()

        # Get processed image
        img = s.image_buffer / 255.0
        network_input = []
        network_input.append(img)
        output = cnn_network.predict(np.array(network_input))
        output[output>0.5] = 1
        output[output<=0.5] = 0
        output = output.flatten()
        print("Action selected:",output)
        action = [int(output[0]), int(output[1]), int(output[2])]
        r = game.make_action(action)

        # Display the image here!
        if game.get_screen_format() in [ScreenFormat.GRAY8, ScreenFormat.DEPTH_BUFFER8]:
            img = img.reshape(img.shape[1],img.shape[2],1)

        cv2.imshow('Doom Buffer',img)
        cv2.waitKey(int(sleep_time * 1000))
        
        #if sleep_time>0:
        #    sleep(sleep_time)

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
