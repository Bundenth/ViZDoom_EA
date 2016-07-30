#!/usr/bin/python
from __future__ import print_function
#from vizdoom import DoomGame
#from vizdoom import Mode
#from vizdoom import Button
#from vizdoom import GameVariable
#from vizdoom import ScreenFormat
#from vizdoom import ScreenResolution
from vizdoom import *


from random import choice,sample
import random
from time import sleep
from time import time
import time

import numpy as np
from six.moves import cPickle

from tqdm import *

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json
import theano.tensor as T

from learning_framework import *
import learning_framework

feature_weights_filename = 'feature_detector_nets/cig_FD_64x48x(4x3x3)_weights.save'
controller_weights_filename = 'full_RL/custom_controller_weights.save'
doom_scenario = "scenarios/custom.wad"
doom_config = "config/custom.cfg"

### general parameters
isTraining = True
use_feature_detector = False #False for full Deep RL; True to use FD
isCig = False

actions_available = [[1,0,0],[0,1,0],[0,0,1],
					[1,0,1],[0,1,1]]

#only for FD RL
num_features = 32 #number of outputs of the CNN compressor (features to learn)

if not use_feature_detector:
	hidden_units = 12 # units in the controller's hidden layer
else:
	hidden_units = 512

# exploration vs exploitation
start_epsilon = float(1.0)
end_epsilon = float(0.1)
epsilon = start_epsilon
static_epsilon_steps = 5000
epsilon_decay_steps = 20000
epsilon_decay_stride = (start_epsilon - end_epsilon) / epsilon_decay_steps

reward_scale = 0.01
replay_memory_size = 10000
discount_factor = 0.99
batch_size = 32
training_steps_per_epoch = 5000
test_episodes_per_epoch = 50
training_epochs = 100

# custom crossentropy
cross_entropy_epsilon = 1.0e-9


##################################
# RL training

# Replay memory:
class ReplayMemory:
    def __init__(self, capacity):
        if(not use_feature_detector):
			state_shape = (capacity, channels, downsampled_y,downsampled_x)
        else:
			state_shape = (capacity, 1, num_features)
		
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.nonterminal = np.zeros(capacity, dtype=np.bool_)

        self.size = 0
        self.capacity = capacity
        self.oldest_index = 0
        
        self.last_shaping_reward = 0
        self.last_ammo = -1
        self.last_health = 100

    def add_transition(self, s1, action, s2, reward):
        if(not use_feature_detector):
			self.s1[self.oldest_index] = s1
        else:
			self.s1[self.oldest_index,0] = s1
        
        if s2 is None:
            self.nonterminal[self.oldest_index] = False
        else:
            if(not use_feature_detector):
				self.s2[self.oldest_index] = s2
            else:
				self.s2[self.oldest_index,0] = s2
            
            self.nonterminal[self.oldest_index] = True
        self.a[self.oldest_index] = action
        self.r[self.oldest_index] = reward

        self.oldest_index = (self.oldest_index + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.s2[i], self.a[i], self.r[i], self.nonterminal[i]


# Makes an action according to epsilon greedy policy and performs a single backpropagation on the network.
def perform_learning_step(game, memory, controller_network,fd_network):
	# Checks the state and downsamples it.
	if(not use_feature_detector):
		s1 = convert(game.get_state().image_buffer)
	else:
		img_p = convert(game.get_state().image_buffer).reshape([1, channels, downsampled_y, downsampled_x])
		s1 = np.reshape(fd_network.predict(img_p).flatten(),(1,num_features))
	
	# With probability epsilon makes a random action.
	if random.random() <= epsilon:
		action = random.randint(0, len(actions_available) - 1)
	else:
		# Get features from image
		#use features to control player
		if(not use_feature_detector):
			actions = controller_network.predict(s1.reshape([1, channels, downsampled_y, downsampled_x])).flatten()
		else:
			actions = controller_network.predict(s1).flatten()
		
		action = np.argmax(actions)
	reward = game.make_action(actions_available[action], skiprate + 1)
	shaping_reward = doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
	reward += shaping_reward - memory.last_shaping_reward
	memory.last_shaping_reward = shaping_reward
	
	if isCig: 
		reward *= 10.0 #kill reward bonus
		ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
		if not memory.last_ammo < 0:
			if ammo > memory.last_ammo:
				reward += (ammo - memory.last_ammo) * 3
			if ammo < memory.last_ammo:
				reward -= 15
		memory.last_ammo = ammo
		health = max(0,game.get_game_variable(GameVariable.HEALTH))
		if health > memory.last_health:
			reward += 30.0
		memory.last_health = health
		if game.is_player_dead():
			reward -= 65.0
	
	reward *= reward_scale
	
	if game.is_episode_finished():
		s2 = None
	else:
		if(not use_feature_detector):
			s2 = convert(game.get_state().image_buffer)
		else:
			img_p = convert(game.get_state().image_buffer).reshape([1, channels, downsampled_y, downsampled_x])
			s2 = np.reshape(fd_network.predict(img_p).flatten(),(1,num_features))
		
	# Remember the transition that was just experienced.
	memory.add_transition(s1, action, s2, reward)

	# Gets a single, random minibatch from the replay memory and learns from it.
	if memory.size > batch_size:
		s1, s2, a, reward, nonterminal = memory.get_sample(batch_size)
		# Only q for the chosen actions is updated more or less according to following formula:
		# target Q(s,a,t) = r + gamma * max Q(s2,_,t+1)
		q2 = []
		for i in range(batch_size):
			if(not use_feature_detector):
				actions = controller_network.predict(s2[i].reshape([1, channels, downsampled_y, downsampled_x])).flatten()
			else:
				actions = controller_network.predict(s2[i]).flatten()
			
			action = np.argmax(actions)
			max_reward = actions[action]
			q2.append(max_reward)
		expected = reward + discount_factor * nonterminal * q2
		state_shape = (batch_size, len(actions_available))
		exp_val = np.zeros(state_shape, dtype=np.float32)
		for i in range(batch_size):
			exp_val[i][a[i]] = expected[i]
		if(not use_feature_detector):
			hist = controller_network.fit(s1,exp_val,verbose=False, batch_size=batch_size, nb_epoch=1)
		else:
			hist = controller_network.fit(s1.reshape([batch_size,num_features]),exp_val,verbose=False, batch_size=batch_size, nb_epoch=1)
		
		loss = hist.history['loss'][-1]
	else:
		loss = 0
	return loss

def custom_loss(y_true, y_pred):
	for i in range(len(actions_available)):
		if(y_true[i] == 0):
			y_true[i] = y_pred[i];
	loss = (y_pred-y_true) ** 2
	return loss
	'''Just another crossentropy'''
	#y_pred = T.clip(y_pred, cross_entropy_epsilon, 1.0 - cross_entropy_epsilon)
	#y_pred /= y_pred.sum(axis=-1, keepdims=True)
	#cce = T.nnet.categorical_crossentropy(y_pred, y_true)
	#return cce

# use Deep RL to train controller
def train_controller(fd_weights_filename,controller_weights_filename):
	epsilon = start_epsilon

	game = DoomGame()
	CustomDoomGame(game,doom_scenario,doom_config)
	start_game(game,isCig,False)

	#load feature detector network
	if not use_feature_detector:
		fd_network = None
	else:
		fd_network = create_cnn(downsampled_y,downsampled_y,num_features)
		fd_network.load_weights(fd_weights_filename)

	#create network controller
	if(not use_feature_detector):
		controller_network = Sequential([
			Convolution2D(32, 8, 8, border_mode='valid',input_shape=(channels, downsampled_y, downsampled_x)),
			Activation('relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Convolution2D(64, 4, 4),
			Activation('relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Convolution2D(64, 3, 3),
			Activation('relu'),
			MaxPooling2D(pool_size=(2, 2)),
			Flatten(),
			Dense(hidden_units),
			Activation('relu'),
			Dense(len(actions_available)),
			Activation('tanh')
		])
	else:
		controller_network = Sequential([
			Dense(hidden_units, input_dim=num_features),
			Activation('tanh'),
			Dense(len(actions_available)),
			Activation('tanh')		
		])
	
	#custom loss function only considering best action
	#http://keras.io/objectives/
	controller_network.compile(loss=custom_loss, #'categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
	# Creates replay memory which will store the transitions
	memory = ReplayMemory(capacity=replay_memory_size)

	# train using reinforcement learning (based on rewards)
	steps = 0
	for i in range(training_epochs):
		print("Episode ",i+1)
		print ("RL Training...")
		train_time = time.time()
		train_episodes_finished = 0
		train_loss = []
		train_rewards = []
		game.new_episode()
		for learning_step in tqdm(range(training_steps_per_epoch)):
			# Learning and action is here.
			train_loss.append(perform_learning_step(game,memory,controller_network,fd_network))
			if game.is_episode_finished():
				r = game.get_total_reward()
				train_rewards.append(r)
				game.new_episode()
				memory.last_shaping_reward = 0
				memory.last_ammo = -1
				memory.last_health = 100
				train_episodes_finished += 1

			steps += 1
			if steps > static_epsilon_steps:
				epsilon = max(end_epsilon, epsilon - epsilon_decay_stride)
		train_time = time.time() - train_time
		mean_loss = np.mean(train_loss)
		train_rewards = np.array(train_rewards)
		print ("mean:", train_rewards.mean(), "std:", train_rewards.std(), "max:", train_rewards.max(), "min:", train_rewards.min(), "mean_loss:", mean_loss, "epsilon:", epsilon)
		print ("t:", str(round(train_time, 2)) + "s")

		print ("Testing...")
		test_episode = []
		test_rewards = []
		test_time = time.time()
		for test_episode in tqdm(range(test_episodes_per_epoch)):
			game.new_episode()
			reward = 0
			while not game.is_episode_finished():
				img_p = convert(game.get_state().image_buffer).reshape([1, channels, downsampled_y, downsampled_x])
				if(not use_feature_detector):
					actions = controller_network.predict(img_p).flatten()
				else:
					features = np.reshape(fd_network.predict(img_p).flatten(),(1,num_features))
					actions = controller_network.predict(features).flatten()
				
				best_action = np.argmax(actions)
				game.make_action(actions_available[best_action],skiprate+1)
				if isCig and game.is_player_dead():
					game.respawn_player()
					reward -= 8.0
			reward += game.get_total_reward()
			test_rewards.append(reward)

		test_time = time.time() - test_time
		print ("Test results:")
		test_rewards = np.array(test_rewards)
		print ("mean:", test_rewards.mean(), "std:", test_rewards.std(), "max:", test_rewards.max(), "min:", test_rewards.min())
		print ("t:", str(round(test_time, 2)) + "s")

		#save solution so far
		controller_network.save_weights(controller_weights_filename,overwrite=True)
	
	game.close()

######################################################################
######################################################################

if isTraining:
	train_controller(feature_weights_filename,controller_weights_filename)


#test
print("Watching")
game = DoomGame()
CustomDoomGame(game,doom_scenario,doom_config)
start_game(game,isCig,True)

if(not use_feature_detector):
	controller_network = Sequential([
		Convolution2D(32, 8, 8, border_mode='valid',input_shape=(channels, downsampled_y, downsampled_x)),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Convolution2D(64, 4, 4),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Convolution2D(64, 3, 3),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Flatten(),
		Dense(hidden_units),
		Activation('relu'),
		Dense(len(actions_available)),
		Activation('tanh')
	])
else:
	controller_network = Sequential([
		Dense(hidden_units, input_dim=num_features),
		Activation('tanh'),
		Dense(len(actions_available)),
		Activation('tanh')		
	])

#custom loss function only considering best action
#http://keras.io/objectives/
controller_network.compile(loss=custom_loss, #'categorical_crossentropy',
	optimizer='adadelta',
	metrics=['accuracy'])
controller_network.load_weights(controller_weights_filename)

if not use_feature_detector:
	fd_network = None
else:
	fd_network = create_cnn(45,60,num_features)
	fd_network.load_weights(feature_weights_filename)

for i in range(100):
	game.new_episode()
	while not game.is_episode_finished():
		img = convert(game.get_state().image_buffer).reshape([1, channels, downsampled_y, downsampled_x])
		if(not use_feature_detector):
			actions = controller_network.predict(img).flatten()
		else:
			features = np.reshape(fd_network.predict(np.array(img)).flatten(),(1,num_features))
			actions = controller_network.predict(features).flatten()
		
		best_action = np.argmax(actions)
		game.make_action(actions_available[best_action])
		
		sleep(0.028)
	r = game.get_total_reward()
	print ("Total reward: ", r)

game.close()
