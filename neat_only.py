#!/usr/bin/python
from __future__ import print_function
from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
# Or just use from vizdoom import *

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
doom_scenario = "scenarios/cig_orig_pistol.wad"
doom_config = "config/cig.cfg"
stats_file = "_stats.txt"

isTraining = False
isCig = True

#needs further downsampling to make it feasible
downsampled_x = 44 #64
downsampled_y = 32#48

reward_multiplier = 5
shoot_reward = -35.0
health_kit_reward = 75.0 #75.0
harm_reward = 0
ammo_pack_reward = 50.0 #50.0

number_actions = 3 #axis + shoot
actions_available = 4
input_dead_zone = 0.2

test_fitness_episodes = 2
epochs = 1000

initial_health = 99

#HyperNEAT parameters
inputs = []
hidden = []
for c in range(channels):
	for i in range(downsampled_y):
		for j in range(downsampled_x):
			inputs += [(i-(downsampled_y/2.),j-(downsampled_x/2.),c-channels/2.)]
inputs += [(1,0,-1.),(0,0,-1.),(-1,0,-1.)]

#NEAT parameters and initialisation
params = NEAT.Parameters()
params.PopulationSize = 100#
# dist = c1*E/N + c2*D/N + c3*W
# E -> excess; D = disjoint; W -> average weight difference
params.DynamicCompatibility = True #
params.CompatTreshold = 15.0 #
params.DisjointCoeff = 1.0#
params.ExcessCoeff = 1.0#
params.WeightDiffCoeff = 0.2#
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


def getAction(net,inp):
	net.Flush()
	net.Input(inp.tolist())
	[net.Activate() for _ in range(4)]
	output = net.Output()
	action = [0 for _ in range(actions_available)]
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


def evaluate(genome):
	net = NEAT.NeuralNetwork()
	reward = 0
	try:
		genome.BuildPhenotype(net)
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
					#reward += death_reward 
					break
			reward += game.get_total_reward() * reward_multiplier
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
		len(inputs),
		0,
		number_actions,
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
		genome_list[best_index].BuildPhenotype(net)
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
	gens = []
	for run in range(10):
		gen = getbest(run)
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
