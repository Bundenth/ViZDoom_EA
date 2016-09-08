#!/usr/bin/python
from __future__ import print_function
from vizdoom import *

from six.moves import cPickle
from random import choice
from time import sleep

from learning_framework import *
import learning_framework

# supports gathering 2 types of images simultaneously
# and from more than one map
images_filename_1 = "feature_images/cig_orig_pistol_marine_hires_rgb.dat"
images_filename_2 = "feature_images/cig_orig_pistol_marine_hires_gray.dat"
doom_scenario = "scenarios/cig_orig_pistol.wad"
doom_config = "config/cig_playable.cfg"
map1 = "map01"
map2 = "map01"

isColourCorrection_1 = False
isColourCorrection_2 = False
channels_1 = 3
channels_2 = 1

isCig = True

recorded_episodes = 6

### FUNCTIONS
# Gather image from play
def gatherData(training_img_set_1,training_img_set_2,mapSelected):
	#configure doom game
	game = DoomGame()
	CustomDoomGame(game,doom_scenario,doom_config,mapSelected)
	start_game(game,isCig,True,Mode.SPECTATOR)
	
	for j in range(recorded_episodes):
		print("Episode #", j+1)
		game.new_episode()
		while not game.is_episode_finished():
			if isCig and game.is_player_dead():
				game.respawn_player()
			# Gets the state
			s = game.get_state()
			sleep(0.005)
			game.advance_action(skiprate+1)
			action = game.get_last_action()
			
			#if not s.number % 8 == 0 or all(v==0 for v in action):
			#	continue
			if action[0] == 0:
				continue
			# Get processed image
			img_1 = learning_framework.convert(s.image_buffer,isColourCorrection_1,channels_1) # [channel][rows][cols]
			training_img_set_1.append(img_1)
			if not images_filename_1 == images_filename_2:
				img_2 = learning_framework.convert(s.image_buffer,isColourCorrection_2,channels_2) # [channel][rows][cols]
				training_img_set_2.append(img_2)
			
			print("new")
		print("Episode finished: ",j+1)
		
		
		
	game.close()
	# End data gathering
	# store data in format: http://deeplearning.net/tutorial/gettingstarted.html
	# create training data as numpy arrays of input and labels
	print("Data gathering finished.")
	print("Total training datapoints:", len(training_img_set_1))
	print("************************")
	# pickle (record) data
	f = open(images_filename_1,'wb')
	cPickle.dump(training_img_set_1, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()
	if not images_filename_1 == images_filename_2:
		f = open(images_filename_2,'wb')
		cPickle.dump(training_img_set_2, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

######################################################################
######################################################################

#gather training set to use in evolution of Feature extractor
gatherData([],[],map1)

if not map1 == map2:
	f = open(images_filename_1,'rb')
	training_img_set_1 = cPickle.load(f)
	f.close()
	training_img_set_2 = []
	if not images_filename_1 == images_filename_2:
		f = open(images_filename_2,'rb')
		training_img_set_2 = cPickle.load(f)
		f.close()
	
	gatherData(training_img_set_1,training_img_set_2,map2)



cv2.destroyAllWindows()

