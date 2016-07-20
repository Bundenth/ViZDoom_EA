#!/usr/bin/python
from __future__ import print_function
from vizdoom import *

from random import choice,sample
import random
from time import sleep
from time import time
import time
import math

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
feature_detector_file = 'feature_detector_nets/cig_orig_pistol_cacodemon_FD_64x48x32_weights.save'
controller_network_filename = 'controller_nets/cig_orig_pistol_cacodemon_32_NEAT_controller.net'
doom_scenario = "scenarios/cig_orig_pistol.wad"
doom_config = "config/cig.cfg"
stats_file = "stats/controller_cig_orig_pistol_cacodemon_32_stats.txt"

num_features = 32
num_states = 1

isTraining = True
isCig = True # whether or not the scenario is competition (cig)
isNEAT = True # choose between NEAT or ES-HyperNEAT
useShapingReward = False

reward_multiplier = 5
shoot_reward = -50.0
health_kit_reward = 75.0 #75.0
harm_reward = 0
ammo_pack_reward = 50.0 #50.0

initial_health = 99

# left,right, forward, backward and shoot and pair-combinations (cig)
#actions_available = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1], #single actions
#			[1,0,1,0,0],[0,1,1,0,0],[1,0,0,1,0],[0,1,0,1,0], #let-right,forward double actions
#			[1,0,0,0,1],[0,1,0,0,1],[0,0,1,0,1],[0,0,0,1,1]] #single actions+shoot
#left, right,forward, backward and pair-combinations (health_gathering)
#actions_available = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1], #single actions
#			[1,0,1,0], [0,1,1,0], [1,0,0,1], [0,1,0,1]] #let-right,forward,backward double actions
# left, right, shoot and pair-combinations (defend_centre)
#actions_available = [[1,0,0],[0,1,0],[0,0,1],
#				[1,0,1],[0,1,1]]
actions_available = 4
number_actions = 3 #axis + shoot
number_axis = 2
number_single_actions = 1
input_dead_zone = 0.2
# left, right, forward backward and pair-combinations (health_gather)
#actions_available = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1]]

test_fitness_episodes = 2
epochs = 1000

#load feature detector network
fd_network = create_cnn(downsampled_y,downsampled_x,num_features)
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
params.YoungAgeFitnessBoost = 1.35
params.SpeciesMaxStagnation = 20 #number of generations without improvement allowed for species
params.OldAgeTreshold = 40# 
params.MinSpecies = 2 #
params.MaxSpecies = 8 #
params.EliteFraction = 0.1 #
params.RouletteWheelSelection = False 
params.CrossoverRate = 0.70#
params.InterspeciesCrossoverRate = 0.001#
params.MutateRemLinkProb = 0.015 
params.RecurrentProb = 0.0015
params.OverallMutationRate = 0.15 
params.MutateAddLinkProb = 0.095#0.05-0.3
params.MutateAddNeuronProb = 0.03 #
params.MutateWeightsProb = 0.85 #
params.WeightMutationMaxPower = 0.5 # 
params.WeightReplacementMaxPower = 1.0 
params.MutateActivationAProb = 0.002 #
params.MutateActivationBProb = 0.002 #
params.ActivationAMutationMaxPower = 0.35 
params.MutateNeuronActivationTypeProb = 0.03 #
params.ActivationFunction_SignedSigmoid_Prob = 1.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
params.ActivationFunction_Tanh_Prob = 1.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 0.0;
params.ActivationFunction_UnsignedStep_Prob = 0.0;
params.ActivationFunction_SignedGauss_Prob = 1.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 0.0;
params.ActivationFunction_SignedSine_Prob = 1.0;
params.ActivationFunction_UnsignedSine_Prob = 0.0;
params.ActivationFunction_Linear_Prob = 1.0;

#HyperNEAT parameters
if not isNEAT:
	substrate_inputs = []
	for i in range(num_states):
		for j in range(num_features):
			substrate_inputs += [(i/(num_states/2.)-1,j/(num_features/2.)-1,-1.)]
	substrate_inputs += [(-2,0,0.),(-1,0,0.),(0,0,0.),(1,0,0.)]
	substrate_outputs = []
	for i in range(number_actions):
		substrate_outputs += [(i/(number_actions/2.)-1,0.,1.)]
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

def start_game(game):
	if isTraining:
		game.set_screen_resolution(ScreenResolution.RES_160X120)
		game.set_window_visible(False)
	else:
		game.set_screen_resolution(ScreenResolution.RES_640X480)
		game.set_window_visible(True)
	
	if isCig:
		# Start multiplayer game only with Your AI (with options that will be used in the competition, details in cig_host example).
		game.add_game_args("-host 1 -deathmatch +timelimit 10.0 "
			"+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
		# Name Your AI.
		game.add_game_args("+name AI")
		game.init()
	else:
		game.init()

def getAction(net,inp):
	net.Flush()
	net.Input(inp.tolist())
	[net.Activate() for _ in range(4)]
	output = net.Output()
	action = [0 for _ in range(actions_available)]
	'''
	current_action = 0
	for axis in range(number_axis):
		if output[axis] > input_dead_zone:
			action[current_action] = 1
		current_action += 1
		if output[axis] < -input_dead_zone:
			action[current_action] = 1
		current_action += 1
	for single_action_ in range(number_single_actions):
		if output[number_axis + single_action_] > input_dead_zone:
			action[current_action] = 1
		current_action += 1
	'''
	if output[0] > input_dead_zone:
		action[0] = 1
	if output[0] < -input_dead_zone:
		action[1] = 1
	if output[1] > input_dead_zone:
		action[2] = 1
	#if output[1] < -input_dead_zone:
	#	action[3] = 1
	if output[2] > input_dead_zone:
		action[3] = 1
	return action


game = DoomGame()
CustomDoomGame(game,doom_scenario,doom_config)
start_game(game)

def getbest(i,controller_network_filename):
	f = open(stats_file,'w')
	f.write("best,average,min,species,neurons,links\n")
	f.close()

	if isNEAT:
		g = NEAT.Genome(0, num_features*num_states+3, 0, number_actions, False, NEAT.ActivationFunction.TANH_CUBIC, 
			NEAT.ActivationFunction.TANH, 0, params)
	else:
		g = NEAT.Genome(0,
			substrate.GetMinCPPNInputs(),
			0,
			substrate.GetMinCPPNOutputs(),
			False,
			NEAT.ActivationFunction.TANH_CUBIC,
			NEAT.ActivationFunction.TANH,
			0,
			params)
	pop = NEAT.Population(g, params, True, 1.0, i)
	pop.RNG.Seed(i)

	max_score = 0
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
		f = open(stats_file,'a')
		f.write(str(best) + ',' + str(avg) + ',' + str(worse) + ',' + str(len(pop.Species)) + ',' + str(genome_list[best_index].NumNeurons()) + ',' + str(genome_list[best_index].NumLinks()) + str("\n"))
		f.close()

		if best > max_score:
			max_score = best
		net.Save(controller_network_filename)

		pop.Epoch()
		generations = generation

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
		counter = 0
		ammo_reward = 0
		health_reward = 0
		shaping_reward = 0
		for ep in range(test_fitness_episodes):
			states = [None for _ in range(num_states)]
			last_ammo = -1
			last_health = initial_health
			try:
				game.new_episode()
			except Exception as ex:
				print('Exception:', ex)
				raise SystemExit
			while not game.is_episode_finished():
				# Get processed image
				counter = counter + 1
				s = game.get_state()
				ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
				health = max(0,game.get_game_variable(GameVariable.HEALTH))
				img = convert(s.image_buffer)
				img = img.reshape([1, channels, downsampled_y, downsampled_x])
				features = fd_network.predict(img).flatten()
				ammo_input = min(float(ammo)/40.0,1.0)
				health_input = float(health)/100.0
				#multistate
				for state in range(num_states-1):
					states[state] = states[state+1] 
				states[num_states-1] = features
				if None in states:
					continue
				inp = np.array(states)
				inp = np.append(inp,[ammo_input,health_input,1.])
				action = getAction(net,inp)
				game.make_action(action,skiprate+1)
				#action = np.argmax(output)
				#game.make_action(actions_available[action],skiprate+1)
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
				if game.is_player_dead():
					#reward += death_reward 
					break
			# use shaping rewards to reinforce behaviours
			if useShapingReward:
				reward += doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
			reward += game.get_total_reward() * reward_multiplier
			#if counter * (skiprate + 1) > training_length:
			#	break
		reward += ammo_reward + health_reward
		return reward

	except Exception as ex:
		print('Exception:', ex)
		return reward


#train
if isTraining:
	gens = []
	for run in range(1):
		gen = getbest(run,controller_network_filename)
		gens += [gen]
		print('Run:', run, 'Max score in DOOM:', gen)
	avg_gens = sum(gens) / len(gens)

	print('All:', gens)
	print('Average:', avg_gens)


#test
net = NEAT.NeuralNetwork()
net.Load(controller_network_filename)

for ep in range(100):
	game.new_episode()
	counter = 0
	states = [None for _ in range(num_states)]
	while not game.is_episode_finished():
		s = game.get_state()
		counter = counter + 1
		ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
		health = max(0,game.get_game_variable(GameVariable.HEALTH))
		ammo_input = min(float(ammo)/40.0,1.0)
		health_input = float(health)/100.0
		img = convert(s.image_buffer)
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
		game.make_action(action,0+1)
		sleep(0.028)
		if game.is_player_dead():
			break

	print(game.get_total_reward())

game.close()

######################################################################
######################################################################



