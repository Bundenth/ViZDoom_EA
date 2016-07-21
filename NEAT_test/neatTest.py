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
from time import sleep

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

downsampled_x = 48
downsampled_y = int(3/4.0*downsampled_x)
view_image = False
skiprate = 3

# Function for converting images
def convert(img):
    img = img[0].astype(np.float32) / 255.0
    img = cv2.resize(img, (downsampled_x, downsampled_y))
    return img

# DOOM parameters
game = DoomGame()
game.set_vizdoom_path("../../../bin/vizdoom")
game.set_doom_game_path("../../../scenarios/doom2.wad")
game.set_doom_scenario_path("../../../scenarios/custom.wad")
game.set_doom_map("map01")
game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(False)
game.set_render_decals(False)
game.set_render_particles(False)
game.set_mode(Mode.PLAYER)
game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.ATTACK)
game.add_available_button(Button.MOVE_RIGHT)
#game.add_available_button(Button.TURN_LEFT)
#game.add_available_button(Button.TURN_RIGHT)
#game.add_available_button(Button.STRAFE)
game.set_episode_timeout(100)
game.set_screen_resolution(ScreenResolution.RES_160X120)
game.set_screen_format(ScreenFormat.GRAY8)
game.set_window_visible(True)
game.set_episode_start_time(10)
game.set_living_reward(-1)

#HyperNEAT parameters
inputs = []
hidden = []
for i in range(downsampled_y):
	for j in range(downsampled_x):
		inputs += [(i/(downsampled_y/2)-1,j/(downsampled_x/2)-1,-1.)]
inputs += [(0,0,-1.)]
#for i in range(downsampled_y/6):
#	for j in range(downsampled_x/6):
#		hidden += [(i/(downsampled_y/12)-1,j/(downsampled_x/12)-1,0.)]
substrate = NEAT.Substrate(inputs,
                           hidden,
                           [(1., -1., 1.),(1., 0., 1.),(1., 1., 1.)])
substrate.m_allow_hidden_hidden_links = True 
substrate.m_allow_output_hidden_links = False 
substrate.m_allow_looped_hidden_links = False 
substrate.m_allow_looped_output_links = False 
substrate.m_allow_input_hidden_links = True 
substrate.m_allow_input_output_links = False# 
substrate.m_allow_hidden_output_links = True
substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID 
substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID 
substrate.m_with_distance = False 
substrate.m_max_weight_and_bias = 8.0# 

params = NEAT.Parameters()
params.PopulationSize = 750#
# dist = c1*E/N + c2*D/N + c3*W
# E -> excess; D = disjoint; W -> average weight difference
params.DynamicCompatibility = True #
params.CompatTreshold = 4.0 #
params.DisjointCoeff = 1.0#
params.ExcessCoeff = 1.0#
params.WeightDiffCoeff = 0.4#
params.YoungAgeTreshold = 5 #fitness multiplier for young species
params.SpeciesMaxStagnation = 15 #number of generations without improvement allowed for species
params.OldAgeTreshold = 35# 
params.MinSpecies = 2 #
params.MaxSpecies = 15 #
params.EliteFraction = 0.1 #
params.RouletteWheelSelection = False 
params.CrossoverRate = 0.75#
params.InterspeciesCrossoverRate = 0.001#
params.MutateRemLinkProb = 0.02 
params.RecurrentProb = 0 
params.OverallMutationRate = 0.15 
params.MutateAddLinkProb = 0.15#0.05-0.3
params.MutateAddNeuronProb = 0.05 #
params.MutateWeightsProb = 0.85 #
params.MaxWeight = 8.0 #
params.WeightMutationMaxPower = 0.35 # 
params.WeightReplacementMaxPower = 1.0 
params.MutateActivationAProb = 0.0 #
params.MutateActivationBProb = 0.0 #
params.ActivationAMutationMaxPower = 0.5 
params.MutateNeuronActivationTypeProb = 0.03 #
params.MinActivationA = 0.05 
params.MaxActivationA = 6.0 
params.ActivationFunction_SignedSigmoid_Prob = 0.0;
params.ActivationFunction_UnsignedSigmoid_Prob = 0.0;
params.ActivationFunction_Tanh_Prob = 1.0;
params.ActivationFunction_TanhCubic_Prob = 0.0;
params.ActivationFunction_SignedStep_Prob = 1.0;
params.ActivationFunction_UnsignedStep_Prob = 0.0;
params.ActivationFunction_SignedGauss_Prob = 1.0;
params.ActivationFunction_UnsignedGauss_Prob = 0.0;
params.ActivationFunction_Abs_Prob = 0.0;
params.ActivationFunction_SignedSine_Prob = 1.0;
params.ActivationFunction_UnsignedSine_Prob = 0.0;
params.ActivationFunction_Linear_Prob = 1.0;

params.DivisionThreshold = 0.5;
params.VarianceThreshold = 0.03;
params.BandThreshold = 0.3;
params.InitialDepth = 2;
params.MaxDepth = 3;
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

def evaluate(genome):
	net = NEAT.NeuralNetwork()
	try:
		genome.BuildESHyperNEATPhenotype(net, substrate, params)
		#genome.BuildHyperNEATPhenotype(net, substrate)
		# do stuff and return the fitness
		reward = 0
		varied_output = False
		for ep in range(2):
			game.new_episode()
			output_read = []
			while not game.is_episode_finished():

				s = game.get_state()

				# Get processed image
				img = s.image_buffer
				img_c = convert(img)
				inp = np.array(img_c).flatten()
				inp = np.append(inp,[1.])
				net.Flush()
				net.Input(inp.tolist())
				[net.Activate() for _ in range(3)]
				output = net.Output()
				#if(len(output_read) > 0):
				#	if(set(output) != set(output_read)):
				#		varied_output = True
				#output_read = [output[0],output[1],output[2]]
				for i in range(3):
					if(output[i] > 0.75):
						output[i] = 1
					else:
						output[i] = 0
				action = [int(output[0]), int(output[1]), int(output[2])]
				r = game.make_action(action,skiprate)
				# Display the image here!
				if(view_image):
					#img[:,:,0] = 0
					#img[:,:,1] = 0
					#img = cv2.resize(img, (320, 240))
					#cv2.imshow('Doom Buffer',img)
					img_c = cv2.resize(img_c, (320, 240))
					cv2.imshow('Doom Buffer',img_c)
					cv2.waitKey(1)

			reward += game.get_total_reward()
		#if(varied_output):
		#	reward += abs(reward)*0.5

		return reward

	except Exception as ex:
		print('Exception:', ex)
		return 1.0

def getbest(i):
	g = NEAT.Genome(0,
		substrate.GetMinCPPNInputs(),
		len(hidden),
		substrate.GetMinCPPNOutputs(),
		False,
		NEAT.ActivationFunction.SIGNED_SIGMOID,
		NEAT.ActivationFunction.UNSIGNED_SIGMOID,
		0,
		params)

	pop = NEAT.Population(g, params, True, 3.0, i)
	pop.RNG.Seed(i)

	max_score = 0
	for generation in range(1000):
		genome_list = NEAT.GetGenomeList(pop)
		fitnesses = EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
		[genome.SetFitness(fitness) for genome, fitness in zip(genome_list, fitnesses)]

		print('Gen: %d Best: %3.5f' % (generation, max(fitnesses)))

		best = max(fitnesses)
		print("Total average score: ",sum(fitnesses) / params.PopulationSize)
		
		#getting information about the generation champion
		best_index = fitnesses.index(best)
		print("Best index: ",best_index,"; Genome details: ", genome_list[best_index].NumNeurons(),genome_list[best_index].NumLinks())
		net = NEAT.NeuralNetwork()
		genome_list[best_index].BuildESHyperNEATPhenotype(net, substrate, params)
		print("Network connections: ",net.GetTotalConnectionLength())
		print("Species: ",len(pop.Species))

		pop.Epoch()
		generations = generation
		net.Save("evolved_net.net")

		if best > max_score:
			max_score = best

	return max_score


game.init()

gens = []
for run in range(1):
    gen = getbest(run)
    gens += [gen]
    print('Run:', run, 'Max score in DOOM:', gen)
avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)

game.close()
cv2.destroyAllWindows()
