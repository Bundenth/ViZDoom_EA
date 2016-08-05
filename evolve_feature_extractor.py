#!/usr/bin/python
from __future__ import print_function
from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
# Or just use from vizdoom import *

from random import choice,sample
import random
import math
from time import sleep
from time import time
import time
import copy

import numpy as np
from six.moves import cPickle

import scipy

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json

from learning_framework import *
import learning_framework

### general parameters
feature_weights_filename = 'feature_detector_nets/cig_orig_pistol_marine_rgb_FD_64x48x32_weights.save'
images_filename = "feature_images/cig_orig_pistol_marine_rgb.dat"
stats_file = "stats/feature_extractor_cig_orig_pistol_marine_rgb_32_stats.txt"

isRandom = False # whether the network generated is randomised or evolved
use_normalisation = True

mutation_rate = 0.0005 #0.003 probability of mutation (prob PER element)
mutation_probability = 0.20 #probability that elite individual is mutated
mutation_strength = 3.0 #1 limit on how much to change a mutated element
novelty_mutation_rate = 0.0001 # mutation that randomises all weights in a sublayer (prob per layer)
weight_start = 5.0

population_size = 100
generations = 1000 #number of generations in the evolution process
num_features = 32 #number of outputs of the CNN compressor (features to learn)
elite_ratio = 0.05 #proportion of top individuals that go to next generation

### FUNCTIONS

# evolution of Feature extractor
def crossover(parent_a,parent_b):
	loop = range(len(parent_a))
	for i in loop:
		if(random.random() < 0.5):
			parent_a[i] = copy.deepcopy(parent_b[i])
	'''
	if(type(parent_a) is np.ndarray or type(parent_a) is list):
		for i in range(len(parent_a)):
			if(type(parent_a[i]) is np.ndarray or type(parent_a[i]) is list):
				crossover(parent_a[i],parent_b[i])
			else:
				if(random.random() < 0.5):
					parent_a[i] = parent_b[i]
	'''


def mutation(offspring):
	if(type(offspring) is np.ndarray or type(offspring) is list):
		for i in range(len(offspring)):
			if(type(offspring[i]) is np.ndarray or type(offspring[i]) is list):
				if(random.random() < novelty_mutation_rate):
					randomise_weights(offspring[i])
					#print("random")
				else:
					mutation(offspring[i])
			else:
				if(random.random() < mutation_rate):
					offspring[i] += random.uniform(-mutation_strength,mutation_strength)

def randomise_weights(cnn):
	if(type(cnn) is np.ndarray or type(cnn) is list):
		for i in range(len(cnn)):
			if(type(cnn[i]) is np.ndarray or type(cnn[i]) is list):
				randomise_weights(cnn[i])
			else:
				cnn[i] = random.uniform(-weight_start,weight_start)

def evaluate(cnn,individual,training_img_set):
	#load weights
	for i in range(len(cnn.layers)):
		cnn.layers[i].set_weights(individual[i])

	#evaluate individual
	feature_vectors = []
	for img in training_img_set:
		# Get processed image
		img_p = img.reshape([1, channels, downsampled_y, downsampled_x])
		output = cnn.predict(np.array(img_p))
		output = output.flatten()
		#Normalized Data
		if use_normalisation:
			magnitude = math.sqrt(sum(output[i]*output[i] for i in range(len(output))))
			normalised = [ output[i]/magnitude  for i in range(len(output)) ]
			feature_vectors.append(normalised)
		else:
			feature_vectors.append(output)

	#calculate fitness
	#approximation of euclidean distances (it's counting distances to i and i points)
	dist = scipy.spatial.distance.cdist(feature_vectors,feature_vectors,'euclidean')
	iu = np.triu_indices_from(dist,1)
	distances = dist[iu]
	fitness = np.mean(distances) + min(distances)
	
	return fitness


# fitness: measure of diversity (min(D) + mean(D))
def evolve_feature_extractor(training_data_filename,weights_filename):
	f = open(stats_file,'w')
	f.write("best,average,worse\n")
	f.close()
	
	print("Loading training data...")
	f = open(training_data_filename,'rb')
	training_img_set = cPickle.load(f)
	f.close()
	
	print("Generating CNN...")
	cnn = create_cnn(downsampled_y,downsampled_x,num_features)

	print("Creating initial population...")
	population = []
	for i in range(population_size):
		population.append([])
		for layer in cnn.layers:
			weights = copy.deepcopy(layer.get_weights())
			randomise_weights(weights)
			population[i].append(weights)
	print("Starting evolution...")
	for gen in range(generations):
		fitnesses = []
		print("Fitness")
		t1 = time.time() 
		for individual in population:
			#evaluate individuals
			fitness = evaluate(cnn,individual,training_img_set)
			fitnesses.append(fitness)
		#sort based on fitness
		indices = range(len(fitnesses))
		fitnesses, indices = (list(t) for t in zip(*sorted(zip(fitnesses,indices),reverse=True)))
		print(time.time()-t1)
		#add elite to new generation
		print("Adding elite")
		new_generation = []
		for i in range(int(population_size * elite_ratio)):
			ind = copy.deepcopy(population[indices[i]])
			if(random.random() < mutation_probability):
				mutation(ind)
			new_generation.append(ind)
		#mate based on ranking
		print("Mating")
		t1 = time.time() 
		probs = [0 for x in range(len(indices))]
		min_fitness = fitnesses[len(fitnesses)-1]
		for i in range(len(indices)):
			n = indices[i]
			probs[n] = 1.0/float(i+1) + fitnesses[i] + min_fitness
		norm_probs = [float(i)/sum(probs) for i in probs]
		mating_pool = np.random.choice(len(population),(population_size-len(new_generation))*2,replace=True,p=norm_probs)
		counter = 0
		while len(new_generation) < population_size:
			ind = copy.deepcopy(population[mating_pool[counter]])
			counter += 1
			crossover(ind,population[mating_pool[counter]])
			counter += 1
			#mutate offspring
			if(random.random() < mutation_probability):
				mutation(ind)
			new_generation.append(ind)
		print(time.time()-t1)
		best = max(fitnesses)
		avg = sum(fitnesses)/len(fitnesses)
		worse = min(fitnesses)
		print("Gen; ", gen+1, "; Best: ",best,"; Avg: ",avg,"; Min: ",worse)
		
		#store training stats
		f = open(stats_file,'a')
		f.write(str(best) + ',' + str(avg) + ',' + str(worse) + '\n')
		f.close()
		
		#store best individual
		best = population[indices[0]]
		for i in range(len(cnn.layers)):
			cnn.layers[i].set_weights(best[i])
		cnn.save_weights(weights_filename,overwrite=True)

		del population[:]
		population = new_generation


def storeRandomNetwork():
	cnn = create_cnn(downsampled_y,downsampled_x,num_features)
	for layer in cnn.layers:
		weights = copy.deepcopy(layer.get_weights())
		randomise_weights(weights)
		layer.set_weights(weights)
	cnn.save_weights(feature_weights_filename,overwrite=True)

######################################################################
######################################################################

#evolve weights of Feature extractor (CNN) using Evolution over training set
if isRandom:
	storeRandomNetwork()
else:
	evolve_feature_extractor(images_filename,feature_weights_filename)

#testing

f = open(images_filename,'rb')
training_img_set = cPickle.load(f)
	
cnn = create_cnn(downsampled_y,downsampled_x,num_features)

cnn.load_weights(feature_weights_filename)

#evaluate individual
feature_vectors = [] #[0 for x in range(num_features)]
for img in training_img_set:
	# Get processed image
	img_p = img.reshape([1, channels, downsampled_y, downsampled_x])
	output = cnn.predict(np.array(img_p))
	output = output.flatten()
	#Normalized Data
	if use_normalisation:
		magnitude = math.sqrt(sum(output[i]*output[i] for i in range(len(output))))
		normalised = [ output[i]/magnitude  for i in range(len(output)) ]
		feature_vectors.append(normalised)
	else:
		feature_vectors.append(output)

#calculate fitness
dist = scipy.spatial.distance.cdist(feature_vectors,feature_vectors,'euclidean')
iu = np.triu_indices_from(dist,1)
distances = dist[iu]
fitness = min(distances) + np.mean(distances)

print(min(distances),np.mean(distances),max(distances))

