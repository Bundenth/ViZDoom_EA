#!/usr/bin/python
from __future__ import print_function
from vizdoom import *

from six.moves import cPickle
from random import choice
from time import sleep

from learning_framework import *
import learning_framework

images_filename = "feature_images/cig_orig_pistol_marine_rgb.dat"
doom_scenario = "scenarios/cig_orig_pistol.wad"
doom_config = "config/cig_playable.cfg"
map1 = "map01"
map2 = "map01"

isCig = True
isColourCorrection = False

recorded_episodes = 3

### FUNCTIONS
# Gather image from play
def gatherData(training_img_set,filename,mapSelected):
	#configure doom game
	game = DoomGame()
	CustomDoomGame(game,doom_scenario,doom_config,mapSelected)
	start_game(game,isCig,True)
	
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
			
			#if all(v==0 for v in action[:3]):
			#	continue
			if action[0] == 0:
				continue
			# Get processed image
			# Gray8 shape is not cv2 compliant
			img = learning_framework.convert(s.image_buffer,isColourCorrection) # [channel][rows][cols]
			
			training_img_set.append(img)
			print("new")
		print("Episode finished: ",j+1)
		
		
		
	game.close()
	# End data gathering
	# store data in format: http://deeplearning.net/tutorial/gettingstarted.html
	# create training data as numpy arrays of input and labels
	print("Data gathering finished.")
	print("Total training datapoints:", len(training_img_set))
	print("************************")
	# pickle (record) data
	f = open(filename,'wb')
	cPickle.dump(training_img_set, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

######################################################################
######################################################################

#gather training set to use in evolution of Feature extractor
gatherData([],images_filename,map1)

if not map1 == map2:
	f = open(images_filename,'rb')
	training_img_set = cPickle.load(f)
	f.close()
	
	gatherData(training_img_set,images_filename,map2)



cv2.destroyAllWindows()

