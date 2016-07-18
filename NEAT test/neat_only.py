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

# Function for converting images
def convert(img):
    img = img[0].astype(np.float32) / 255.0
    img = cv2.resize(img, (downsampled_x, downsampled_y))
    return img

# DOOM parameters
game = DoomGame()
game.set_vizdoom_path("../../bin/vizdoom")
game.set_doom_game_path("../../scenarios/doom2.wad")
game.set_doom_scenario_path("../../scenarios/custom.wad")
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

params = NEAT.Parameters()
params.PopulationSize = 150
params.DynamicCompatibility = True
params.WeightDiffCoeff = 4.0
params.CompatTreshold = 2.0
params.YoungAgeTreshold = 15
params.SpeciesMaxStagnation = 15
params.OldAgeTreshold = 35
params.MinSpecies = 5
params.MaxSpecies = 10
params.RouletteWheelSelection = False
params.RecurrentProb = 0.0
params.OverallMutationRate = 0.8
params.MutateWeightsProb = 0.90
params.WeightMutationMaxPower = 2.5
params.WeightReplacementMaxPower = 5.0
params.MutateWeightsSevereProb = 0.5
params.WeightMutationRate = 0.25
params.MaxWeight = 8
params.MutateAddNeuronProb = 0.03
params.MutateAddLinkProb = 0.05
params.MutateRemLinkProb = 0.01
params.MinActivationA  = 4.9
params.MaxActivationA  = 4.9
params.ActivationFunction_SignedSigmoid_Prob = 1.0
params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
params.ActivationFunction_Tanh_Prob = 0.0
params.ActivationFunction_SignedStep_Prob = 0.0
params.CrossoverRate = 0.75  # mutate only 0.25
params.MultipointCrossoverRate = 0.4
params.SurvivalRate = 0.2

def evaluate(genome):
	net = NEAT.NeuralNetwork()
	try:
		genome.BuildPhenotype(net)
		# do stuff and return the fitness
		reward = 0
		for ep in range(2):
			game.new_episode()

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
				output = [(x+1)/2 for x in output]
				for i in range(3):
					if(output[i] > 0.75):
						output[i] = 1
					else:
						output[i] = 0
				action = [int(output[0]), int(output[1]), int(output[2])]
				r = game.make_action(action)
				# Display the image here!
				if(view_image):
					#img[:,:,0] = 0
					#img[:,:,1] = 0
					#img = cv2.resize(img, (320, 240))
					#cv2.imshow('Doom Buffer',img)
					img_c = cv2.resize(img_c, (320, 240))
					cv2.imshow('Doom Buffer',img_c)
					cv2.waitKey(1)
				#if sleep_time>0:
				#    sleep(sleep_time)
			reward += game.get_total_reward()

		return reward

	except Exception as ex:
		print('Exception:', ex)
		return 1.0

def getbest(i):
	g = NEAT.Genome(0,
		len(inputs),
		0,
		3,
		False,
		NEAT.ActivationFunction.TANH,
		NEAT.ActivationFunction.TANH,
		0,
		params)

	pop = NEAT.Population(g, params, True, 1.0, i)
	pop.RNG.Seed(i)

	max_score = 0
	for generation in range(2000):
		genome_list = NEAT.GetGenomeList(pop)
		fitnesses = EvaluateGenomeList_Serial(genome_list, evaluate, display=False)
		NEAT.ZipFitness(genome_list, fitnesses)
        
		print('Gen: %d Best: %3.5f' % (generation, max(fitnesses)))

		best = max(fitnesses)
		print("Total average score: ",sum(fitnesses) / params.PopulationSize)
		
		#getting information about the generation champion
		best_index = fitnesses.index(best)
		print("Best index: ",best_index,"; Genome details: ", genome_list[best_index].NumNeurons(),genome_list[best_index].NumLinks())
		net = NEAT.NeuralNetwork()
		genome_list[best_index].BuildPhenotype(net)
		print("Network connections: ",net.GetTotalConnectionLength())
		pop.Epoch()
		generations = generation

		if best > max_score:
			max_score = best

	return max_score


game.init()

gens = []
for run in range(10):
    gen = getbest(run)
    gens += [gen]
    print('Run:', run, 'Max score in DOOM:', gen)
avg_gens = sum(gens) / len(gens)

print('All:', gens)
print('Average:', avg_gens)

game.close()
cv2.destroyAllWindows()
