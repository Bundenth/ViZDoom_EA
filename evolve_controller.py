#!/usr/bin/python
from __future__ import print_function
from vizdoom import *

from random import choice,sample
import random
from time import sleep
from time import time
import time
import math
import os

import numpy as np
from six.moves import cPickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json
import MultiNEAT as NEAT
from MultiNEAT import GetGenomeList, ZipFitness, EvaluateGenomeList_Serial, EvaluateGenomeList_Parallel

from learning_framework import *
import learning_framework


### general parameters
feature_detector_file = 'feature_detector_nets/random/FD_64x48x16_1.save'
controller_network_filename = 'controller_nets/pursuit_and_gather/NEAT_actionSelection_16_random_sigmoid_1_X2/controller'
test_controller_net_gen = -1 # -1 to record all generations performance, > -1 to test specific generation 
doom_scenario = "scenarios/pursuit_and_gather.wad"
doom_config = "config/pursuit_and_gather.cfg"
stats_file = "_stats.txt"
evaluation_filename = "_eval.txt"
map1 = "map01"
map2 = "map01"

fd_fitness_factor = FD_Fitness_factor.RANDOM
neat_output_activation = NEAT.ActivationFunction.SIGNED_SIGMOID #NEAT.ActivationFunction.LINEAR

num_features = 16
num_states = 1

isTraining = True
slowTestEpisode = False #whether test performance episodes should be slowed down
isCig = False # whether or not the scenario is competition (cig)
isNEAT = True # choose between NEAT or ES-HyperNEAT
isFS_NEAT = False # False: start with all inputs linked to all outputs; True: random input-output links
useShapingReward = False
isColourCorrection = False
useActionSelection = True # whether output units are final actions or each unit forms a part of an action


###parameters set automatically based on FD_Fitness_factor
output_activation_function = 'sigmoid'
binary_threshold = 0.0 # threshold to consider output active (1) or inactive (0). Value of 0 won't use binary thresholding
###

reward_multiplier = 1
shoot_reward = -35.0
health_kit_reward = 75.0 #75.0
harm_reward = 0
ammo_pack_reward = 50.0 #50.0
death_reward = 0.0

initial_health = 100

evaluation_episodes = 2
epochs = 230
test_fitness_episodes = 4


if fd_fitness_factor == FD_Fitness_factor.VECTOR_DISTANCE_TANH:
	output_activation_function = 'tanh'
if fd_fitness_factor == FD_Fitness_factor.VECTOR_DISTANCE_LINEAR:
	output_activation_function = 'linear'
if fd_fitness_factor == FD_Fitness_factor.SHANNON_AVG:
	output_activation_function = 'sigmoid'
	binary_threshold = 0.0
if fd_fitness_factor == FD_Fitness_factor.SHANNON_BINARY:
	output_activation_function = 'sigmoid'
	binary_threshold = 0.5


if useActionSelection:
	# left,right, forward and shoot and pair-combinations (pursuit and gather)
	actions_available = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1], #single actions
				[1,0,1,0],[0,1,1,0], #let-right,forward double actions
				[1,0,0,1],[0,1,0,1],[0,0,1,1]] #single actions+shoot
	#left, right,forward, backward and pair-combinations (health_gathering)
	#actions_available = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1], #single actions
	#			[1,0,1,0], [0,1,1,0], [1,0,0,1], [0,1,0,1]] #let-right,forward,backward double actions
	# left, right, shoot and pair-combinations (defend_centre)
	#actions_available = [[1,0,0],[0,1,0],[0,0,1]]
	#				[1,0,1],[0,1,1]]
else:
	actions_available = 4
	number_actions = 3 #axis + shoot
	input_dead_zone = 1000


#load feature detector network
fd_network = create_cnn(downsampled_y,downsampled_x,num_features,output_activation_function)
fd_network.load_weights(feature_detector_file)

#NEAT parameters and initialisation
params = NEAT.Parameters()
params.PopulationSize = 100#
# dist = c1*E/N + c2*D/N + c3*W
# E -> excess; D = disjoint; W -> average weight difference
params.DynamicCompatibility = True #
params.CompatTreshold = 5.0 #
params.DisjointCoeff = 1.0#
params.ExcessCoeff = 1.0#
params.WeightDiffCoeff = 0.4#
params.YoungAgeTreshold = 5 #fitness multiplier for young species
params.YoungAgeFitnessBoost = 1.5
params.SpeciesMaxStagnation = 30 #number of generations without improvement allowed for species
params.OldAgeTreshold = 50# 
params.MinSpecies = 2 #
params.MaxSpecies = 8 #
params.EliteFraction = 0.1 #
params.RouletteWheelSelection = False 
params.CrossoverRate = 0.70#0.70
params.InterspeciesCrossoverRate = 0.01#
params.MutateRemLinkProb = 0.035
params.RecurrentProb = 0.0015
params.OverallMutationRate = 0.25 #0.15
params.MutateAddLinkProb = 0.09#0.095
params.MutateAddNeuronProb = 0.03 #0.03
params.MutateWeightsProb = 0.85 #
params.WeightMutationMaxPower = 1.0 # 
params.WeightReplacementMaxPower = 1.0 
#params.MutateActivationAProb = 0.002 #
#params.MutateActivationBProb = 0.002 #
#params.ActivationAMutationMaxPower = 0.35 
params.MutateNeuronActivationTypeProb = 0.03 #

params.ActivationFunction_SignedSigmoid_Prob = 0.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
params.ActivationFunction_Tanh_Prob = 0.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 0.0;
params.ActivationFunction_UnsignedStep_Prob = 0.0;
params.ActivationFunction_SignedGauss_Prob = 0.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 0.0;
params.ActivationFunction_SignedSine_Prob = 0.0;
params.ActivationFunction_UnsignedSine_Prob = 0.0;
params.ActivationFunction_Linear_Prob = 0.0;
params.ActivationFunction_Relu_Prob = 1.0;
params.ActivationFunction_Softplus_Prob = 0.0;


#HyperNEAT parameters
if not isNEAT:
	substrate_inputs = []
	for i in range(num_states):
		for j in range(num_features):
			substrate_inputs += [(i/(num_states/2.)-1,j/(num_features/2.)-1,-1.)]
	substrate_inputs += [(-1,0,0.),(0,0,0.),(1,0,0.)]
	substrate_outputs = []
	if useActionSelection:
		output_units = len(actions_available)
	else:
		output_units = number_actions
	for i in range(output_units):
		substrate_outputs += [(i/(output_units/2.)-1,0.,1.)]
	substrate = NEAT.Substrate(substrate_inputs,
							   [],
							   substrate_outputs)
	substrate.m_allow_hidden_hidden_links = True 
	substrate.m_allow_output_hidden_links = True 
	substrate.m_allow_looped_hidden_links = False 
	substrate.m_allow_looped_output_links = False 
	substrate.m_allow_input_hidden_links = True 
	substrate.m_allow_input_output_links = True# 
	substrate.m_allow_hidden_output_links = True
	substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID 
	substrate.m_output_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID 
	substrate.m_with_distance = False 
	#substrate.m_max_weight_and_bias = 8.0# 

	params.DivisionThreshold = 0.5;
	params.VarianceThreshold = 0.03;
	params.BandThreshold = 0.3;
	params.InitialDepth = 2;
	params.MaxDepth = 4;
	params.IterationLevel = 1;
	params.Leo = False;
	params.GeometrySeed = False;
	params.LeoSeed = False;
	params.LeoThreshold = 0.3;
	params.CPPN_Bias = -1.0;
	params.Qtree_X = 0.0;
	params.Qtree_Y = 0.0;
	params.Width = 2.;
	params.Height = 2.;
	params.Elitism = 0.1;


def getAction(net,inp):
	net.Flush()
	net.Input(inp.tolist())
	[net.Activate() for _ in range(4)]
	output = net.Output()
	if useActionSelection:
		action = np.argmax(output)
		return actions_available[action]
	else:
		action = [0 for _ in range(actions_available)]
		if output[0] > input_dead_zone:
			action[0] = 1
		if output[0] < -input_dead_zone:
			action[1] = 1
		if output[1] > input_dead_zone:
			action[2] = 1
		if output[2] > input_dead_zone:
			action[3] = 1
		'''
		if output[0] > 0.5 + input_dead_zone/2.0:
			action[0] = 1
		if output[0] < 0.5 - input_dead_zone/2.0:
			action[1] = 1
		if output[1] > 0.5 + input_dead_zone/2.0:
			action[2] = 1
		if output[1] < 0.5 - input_dead_zone/2.0:
			action[3] = 1
		if output[2] > input_dead_zone:
			action[4] = 1
		'''
		return action


game = DoomGame()
CustomDoomGame(game,doom_scenario,doom_config,map1)
start_game(game,isCig,not isTraining and slowTestEpisode)

if map1 == map2:
	game2 = game
else:
	game2 = DoomGame()
	CustomDoomGame(game2,doom_scenario,doom_config,map2)
	start_game(game2,isCig,not isTraining and slowTestEpisode)

def getbest(i,controller_network_filename):
	if not os.path.exists(os.path.dirname(controller_network_filename)):
		os.makedirs(os.path.dirname(controller_network_filename))
	f = open(os.path.dirname(controller_network_filename) + '/' + stats_file,'w')
	f.write("best,average,min,species,neurons,links\n")
	f.close()

	if isNEAT:
		if useActionSelection:
			output_units = len(actions_available)
		else:
			output_units = number_actions
		g = NEAT.Genome(0,
			num_features*num_states+3, 
			0, 
			output_units, 
			isFS_NEAT, 
			neat_output_activation, # output activation
			NEAT.ActivationFunction.RELU,  # hidden activation
			0, 
			params)
	else:
		g = NEAT.Genome(0,
			substrate.GetMinCPPNInputs(),
			0,
			substrate.GetMinCPPNOutputs(),
			isFS_NEAT,
			neat_output_activation,
			NEAT.ActivationFunction.RELU,
			0,
			params)
	pop = NEAT.Population(g, params, True, 1.0, i)
	pop.RNG.Seed(i)

	max_score = 0
	starting_time = time.time()
	for generation in range(epochs):
		genome_list = NEAT.GetGenomeList(pop)
		fitnesses = EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
		[genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

		best = max(fitnesses)
		avg = sum(fitnesses) / params.PopulationSize
		worse = min(fitnesses)
		print('Gen: %d Best: %d' % (generation+1, best))
		print("Average: ",avg,"; Min: ",worse)
		
		#getting information about the generation champion
		best_index = fitnesses.index(best)
		print("Best index: ",best_index,"; Genome details: ", genome_list[best_index].NumNeurons(),genome_list[best_index].NumLinks())
		net = NEAT.NeuralNetwork()
		if isNEAT:
			genome_list[best_index].BuildPhenotype(net)
		else:
			genome_list[best_index].BuildESHyperNEATPhenotype(net, substrate, params)
		
		print("Species: ",len(pop.Species))
		print("*****")
		#store training stats
		f = open(os.path.dirname(controller_network_filename) + '/' + stats_file,'a')
		f.write(str(best) + ',' + str(avg) + ',' + str(worse) + ',' + str(len(pop.Species)) + ',' + str(genome_list[best_index].NumNeurons()) + ',' + str(genome_list[best_index].NumLinks()) + str("\n"))
		f.close()

		if best > max_score:
			max_score = best
		net.Save(controller_network_filename + str(generation+1) + '.net')

		pop.Epoch()
		generations = generation
	
	# store training time
	f = open(os.path.dirname(controller_network_filename) + '/' + stats_file,'a')
	f.write('traininig time: ' + str(time.time()-starting_time) + str("\n"))
	f.close()

	return max_score


################################
# use NEAT to evolve controller
def evaluate(genome):
	net = NEAT.NeuralNetwork()
	reward = 0
	try:
		if isNEAT:
			genome.BuildPhenotype(net)
		else:
			genome.BuildESHyperNEATPhenotype(net, substrate, params)
		# do stuff and return the fitness
		ammo_reward = 0
		health_reward = 0
		shaping_reward = 0
		for ep in range(evaluation_episodes):
			if ep % 2 == 0:
				g = game
			else:
				g = game2
			states = [None for _ in range(num_states)]
			last_ammo = -1
			last_health = initial_health
			try:
				g.new_episode()
			except Exception as ex:
				print('Exception:', ex)
				raise SystemExit
			error_counter = 0
			none_state_counter = 0
			while not g.is_episode_finished():
				# Get processed image
				s = g.get_state()
				if s.image_buffer is None:
					print("Image was None")
					error_counter += 1
					if error_counter > 2:
						print("Too many None states")
						break
					continue
				img = convert(s.image_buffer,isColourCorrection)
				img = img.reshape([1, channels, downsampled_y, downsampled_x])
				features = fd_network.predict(img).flatten()
				if binary_threshold > 0:
					features[features>binary_threshold] = 1
					features[features<=binary_threshold] = 0
				ammo = g.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
				health = max(0,g.get_game_variable(GameVariable.HEALTH))
				ammo_input = min(float(ammo)/40.0,1.0)
				health_input = min(float(health)/100.0,1.0)
				#multistate
				for state in range(num_states-1):
					states[state] = states[state+1] 
				states[num_states-1] = features
				if None in states:
					none_state_counter += 1
					if none_state_counter > num_states:
						print("Too many None states")
						break
					continue
				inp = np.array(states)
				inp = np.append(inp,[ammo_input,health_input,1.])
				action = getAction(net,inp)
				g.make_action(action,skiprate+1)
				#update variables
				if not last_ammo < 0:
					if ammo > last_ammo:
						ammo_reward = ammo_reward + ammo_pack_reward
					if ammo < last_ammo:
						ammo_reward = ammo_reward + shoot_reward #90.0 / float(ammo+1)
				last_ammo = ammo
				if health > last_health:
					health_reward = health_reward + health_kit_reward
				if health < last_health - 10: #if harm
					health_reward = health_reward + (health - last_health) * harm_reward
				last_health = health
				if g.is_player_dead():
					reward += death_reward 
					break
			# use shaping rewards to reinforce behaviours
			if useShapingReward:
				reward += doom_fixed_to_double(g.get_game_variable(GameVariable.USER1))
			reward += g.get_total_reward() * reward_multiplier
			#if counter * (skiprate + 1) > training_length:
			#	break
		reward += ammo_reward + health_reward
		return reward

	except Exception as ex:
		print('Exception:', ex)
		return reward

def play(net,episodes,game,game2,storeStats):
	reward = 0
	for ep in range(episodes):
		if ep % 2 == 0:
			g = game
		else:
			g = game2
		g.new_episode()
		states = [None for _ in range(num_states)]
		while not g.is_episode_finished():
			s = g.get_state()
			if s.image_buffer is None:
				print("Image was None")
				continue
			ammo = g.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
			health = max(0,g.get_game_variable(GameVariable.HEALTH))
			ammo_input = min(float(ammo)/40.0,1.0)
			health_input = min(float(health)/100.0,1.0)
			img = convert(s.image_buffer,isColourCorrection)
			img = img.reshape([1, channels, downsampled_y, downsampled_x])
			features = fd_network.predict(img).flatten()
			if binary_threshold > 0:
				features[features>binary_threshold] = 1
				features[features<=binary_threshold] = 0
			#multistate
			for state in range(num_states-1):
				states[state] = states[state+1] 
			states[num_states-1] = features
			if None in states:
				continue
			inp = np.array(states)
			inp = np.append(inp,[ammo_input,health_input,1.])
			action = getAction(net,inp)
			doSkip = skiprate
			if not isTraining and slowTestEpisode:
				doSkip = 0
				sleep(0.028)
			g.make_action(action,doSkip+1)
			if g.is_player_dead():
				break
		reward += g.get_total_reward()
	reward = reward / float(episodes)
	print('Reward: ',reward)
	if storeStats:
		#store stats
		f = open(os.path.dirname(controller_network_filename) + '/' + evaluation_filename,'a')
		f.write(str(reward) + ',' + str(episodes) + str("\n"))
		f.close()

#train
if isTraining:
	gen = getbest(int(random.random() * 100),controller_network_filename)
	print('Max score in DOOM:', gen)

#test
if test_controller_net_gen > -1:
	net = NEAT.NeuralNetwork()
	net.Load(controller_network_filename + str(test_controller_net_gen) + '.net')

	play(net,test_fitness_episodes,game,game2,False)
else:
	#store stats
	f = open(os.path.dirname(controller_network_filename) + '/' + evaluation_filename,'w')
	f.write('total reward' + str("\n"))
	f.close()
	for gen in range(epochs):
		net = NEAT.NeuralNetwork()
		net.Load(controller_network_filename + str(gen+1) + '.net')
		play(net,test_fitness_episodes,game,game2,True)
	
game.close()
game2.close()

### old
'''
net = NEAT.NeuralNetwork()
net.Load(controller_network_filename + str(test_controller_net_gen) + '.net')

#store stats
f = open(os.path.dirname(controller_network_filename) + '/' + evaluation_filename,'w')
f.write('total reward' + str("\n"))
f.close()

for ep in range(test_fitness_episodes):
	if ep % 2 == 0:
		g = game
	else:
		g = game2
	g.new_episode()
	counter = 0
	states = [None for _ in range(num_states)]
	while not g.is_episode_finished():
		s = g.get_state()
		counter = counter + 1
		ammo = g.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
		health = max(0,g.get_game_variable(GameVariable.HEALTH))
		ammo_input = min(float(ammo)/40.0,1.0)
		health_input = min(float(health)/100.0,1.0)
		img = convert(s.image_buffer,isColourCorrection)
		img = img.reshape([1, channels, downsampled_y, downsampled_x])
		features = fd_network.predict(img).flatten()
		#multistate
		for state in range(num_states-1):
			states[state] = states[state+1] 
		states[num_states-1] = features
		if None in states:
			continue
		inp = np.array(states)
		inp = np.append(inp,[ammo_input,health_input,1.])
		action = getAction(net,inp)
		g.make_action(action,0+1)
		sleep(0.028)
		if g.is_player_dead():
			break
		print(g.get_total_reward())
	#store stats
	f = open(os.path.dirname(controller_network_filename) + '/' + evaluation_filename,'a')
	f.write(str(g.get_total_reward()) + str("\n"))
	f.close()

game.close()
game2.close()
'''
######################################################################
######################################################################



