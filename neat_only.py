#!/usr/bin/python
from __future__ import print_function
from vizdoom import *

from random import choice
import random
from time import sleep
from time import time
import time

import numpy as np
from six.moves import cPickle
import h5py
import cv2

import os
import sys
import subprocess as comm
import MultiNEAT as NEAT
from MultiNEAT import GetGenomeList, ZipFitness, EvaluateGenomeList_Serial, EvaluateGenomeList_Parallel

from concurrent.futures import ProcessPoolExecutor, as_completed

from learning_framework import *
import learning_framework

controller_network_filename = 'NEAT_test/evolved_net.net'
doom_scenario = "scenarios/pursuit_and_gather.wad"
doom_config = "config/pursuit_and_gather.cfg"
stats_file = "_stats.txt"

isTraining = False
isCig = False

#needs further downsampling to make it feasible
#downsampled_x = 64 #64
#downsampled_y = 48#48

reward_multiplier = 1
shoot_reward = -35.0
health_kit_reward = 75.0 #75.0
harm_reward = 0
ammo_pack_reward = 50.0 #50.0

number_actions = 3 #axis + shoot
actions_available = 4
input_dead_zone = 0.2

test_fitness_episodes = 2
epochs = 1000

initial_health = 100

#NEAT parameters and initialisation
params = NEAT.Parameters()
params.PopulationSize = 80#
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

params.ActivationFunction_SignedSigmoid_Prob = 1.0;
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
substrate_inputs = []
for c in range(channels):
	for i in range(downsampled_y):
		for j in range(downsampled_x):
			substrate_inputs += [(i-(downsampled_y/2.),j-(downsampled_x/2.),c-channels/2.)]
substrate_inputs += [(1,0,-1.),(0,0,-1.),(-1,0,-1.)]
substrate_hidden = []
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
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.RELU 
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
	[net.Activate() for _ in range(5)]
	output = net.Output()
	action = [0 for _ in range(actions_available)]
	if output[0] > input_dead_zone:
		action[0] = 1
	if output[0] < -input_dead_zone:
		action[1] = 1
	if output[1] > input_dead_zone:
		action[2] = 1
	if output[2] > input_dead_zone:
		action[3] = 1
	return action


def evaluate(genome):
	net = NEAT.NeuralNetwork()
	reward = 0
	try:
		genome.BuildESHyperNEATPhenotype(net, substrate, params)
		# do stuff and return the fitness
		ammo_reward = 0
		health_reward = 0
		for ep in range(test_fitness_episodes):
			last_ammo = -1
			last_health = initial_health
			try:
				game.new_episode()
			except Exception as ex:
				print('Exception:', ex)
				raise SystemExit
			while not game.is_episode_finished():
				# Get processed image
				s = game.get_state()
				ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
				health = max(0,game.get_game_variable(GameVariable.HEALTH))
				img = convert(s.image_buffer)
				#img = img.reshape([1, channels, downsampled_y, downsampled_x])
				ammo_input = min(float(ammo)/40.0,1.0)
				health_input = float(health)/100.0
				inp = np.array(img)
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
					break
			reward += game.get_total_reward() * reward_multiplier + doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
		reward += ammo_reward + health_reward
		return reward

	except Exception as ex:
		print('Exception:', ex)
		return reward

def getbest(i):
	if not os.path.exists(os.path.dirname(controller_network_filename)):
		os.makedirs(os.path.dirname(controller_network_filename))
	f = open(os.path.dirname(controller_network_filename) + '/' + stats_file,'w')
	f.write("best,average,min,species,neurons,links\n")
	f.close()

	g = NEAT.Genome(0,
		substrate.GetMinCPPNInputs(),
		len(substrate_hidden),
		substrate.GetMinCPPNOutputs(),
		False,
		NEAT.ActivationFunction.SIGNED_SIGMOID,
		NEAT.ActivationFunction.RELU,
		0,
		params)

	pop = NEAT.Population(g, params, True, 1.0, i)
	pop.RNG.Seed(i)

	max_score = 0
	for generation in range(epochs):
		genome_list = NEAT.GetGenomeList(pop)
		fitnesses = EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
		NEAT.ZipFitness(genome_list, fitnesses)
        
		best = max(fitnesses)
		avg = sum(fitnesses) / params.PopulationSize
		worse = min(fitnesses)
		print('Gen: %d Best: %d' % (generation+1, best))
		print("Average: ",avg,"; Min: ",worse)
		
		#getting information about the generation champion
		best_index = fitnesses.index(best)
		print("Best index: ",best_index,"; Genome details: ", genome_list[best_index].NumNeurons(),genome_list[best_index].NumLinks())
		net = NEAT.NeuralNetwork()
		genome_list[best_index].BuildESHyperNEATPhenotype(net, substrate, params)
		print("Species: ",len(pop.Species))
		print("*****")
		
		#store training stats
		f = open(os.path.dirname(controller_network_filename) + '/' + stats_file,'a')
		f.write(str(best) + ',' + str(avg) + ',' + str(worse) + ',' + str(len(pop.Species)) + ',' + str(genome_list[best_index].NumNeurons()) + ',' + str(genome_list[best_index].NumLinks()) + str("\n"))
		f.close()
		
		net.Save(controller_network_filename)
		
		pop.Epoch()
		generations = generation

		if best > max_score:
			max_score = best

	return max_score


game = DoomGame()
CustomDoomGame(game,doom_scenario,doom_config,"map01")
start_game(game,isCig,not isTraining)

if isTraining:
	gen = getbest(int(random.random() * 100))
	print('Max score in DOOM:', gen)

#test
net = NEAT.NeuralNetwork()
net.Load(controller_network_filename)

for ep in range(100):
	game.new_episode()
	counter = 0
	while not game.is_episode_finished():
		s = game.get_state()
		counter = counter + 1
		ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
		health = max(0,game.get_game_variable(GameVariable.HEALTH))
		ammo_input = min(float(ammo)/40.0,1.0)
		health_input = float(health)/100.0
		img = convert(s.image_buffer)
		#img = img.reshape([1, channels, downsampled_y, downsampled_x])
		ammo_input = min(float(ammo)/40.0,1.0)
		health_input = float(health)/100.0
		inp = np.array(img)
		inp = np.append(inp,[ammo_input,health_input,1.])
		action = getAction(net,inp)
		game.make_action(action,0+1)
		sleep(0.028)
		if game.is_player_dead():
			break

	print(game.get_total_reward())

game.close()
