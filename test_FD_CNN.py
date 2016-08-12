#!/usr/bin/python
from __future__ import print_function
from vizdoom import *

from six.moves import cPickle
from random import choice
from time import sleep

from learning_framework import *
import learning_framework

doom_scenario = "scenarios/pursuit_and_gather.wad"
doom_config = "config/pursuit_and_gather.cfg"
mapSelected = "map01"
feature_weights_filename = 'feature_detector_nets/pursuit_and_gather/FD_64x48x32_distanceT'
test_fd_net_gen = '0'

num_features = 32
binary_threshold = 0.0

fd_fitness_factor = FD_Fitness_factor.VECTOR_DISTANCE_LINEAR

isCig = False
episodes = 5


#assign appropriate activation function for output layer
output_activation_function = 'sigmoid'
if fd_fitness_factor == FD_Fitness_factor.VECTOR_DISTANCE_TANH:
	output_activation_function = 'tanh'
if fd_fitness_factor == FD_Fitness_factor.VECTOR_DISTANCE_LINEAR:
	output_activation_function = 'linear'
if fd_fitness_factor == FD_Fitness_factor.SHANNON_AVG:
	output_activation_function = 'sigmoid'
if fd_fitness_factor == FD_Fitness_factor.SHANNON_BINARY:
	output_activation_function = 'sigmoid'


cnn = create_cnn(downsampled_y,downsampled_x,num_features,output_activation_function)
cnn.load_weights(feature_weights_filename + '_' + test_fd_net_gen + '.save')

#configure doom game
game = DoomGame()
CustomDoomGame(game,doom_scenario,doom_config,mapSelected)
start_game(game,isCig,True,Mode.SPECTATOR)

for j in range(episodes):
	print("Episode #", j+1)
	game.new_episode()
	while not game.is_episode_finished():
		if isCig and game.is_player_dead():
			game.respawn_player()
		# Gets the state
		s = game.get_state()
		
		game.advance_action(0+1)
		action = game.get_last_action()
		img_p = convert(s.image_buffer).reshape([1, channels, downsampled_y, downsampled_x])
		output = cnn.predict(np.array(img_p))
		output = output.flatten()
		if binary_threshold > 0:
			output[output>binary_threshold] = 1
			output[output<=binary_threshold] = 0
		print(output[0])
	print("Episode finished: ",j+1)
	
game.close()
######################################################################
######################################################################


