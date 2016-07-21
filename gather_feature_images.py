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
isCig = True
checkDupes = False

recorded_episodes = 6

### FUNCTIONS
# Gather image from play
def gatherData(filename):
	training_img_set = []
	#configure doom game
	game = DoomGame()
	CustomDoomGame(game,doom_scenario,doom_config)
	if isCig:
		# Start multiplayer game only with Your AI (with options that will be used in the competition, details in cig_host example).
		game.add_game_args("-host 1 -deathmatch +timelimit 10.0 "
                   "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
		# Name Your AI.
		game.add_game_args("+name AI")
	game.init()
	for j in range(recorded_episodes):
		print("Episode #", j+1)
		#if isCig:
		#	for i in range(bots_number):
		#		game.send_game_command("addbot")
		#else:
			# Starts a new episode. 
		#	game.new_episode()
		game.new_episode()
		while not game.is_episode_finished():
			if isCig and game.is_player_dead():
				game.respawn_player()
			# Gets the state
			s = game.get_state()
			sleep(0.005)
			game.advance_action(skiprate+1)
			action = game.get_last_action()
			
			if all(v==0 for v in action[:3]):
				#skip frame
				continue
			if action[0] == 0:
				continue
			# Get processed image
			# Gray8 shape is not cv2 compliant
			img = learning_framework.convert(s.image_buffer) # [channel][rows][cols]

			if checkDupes:# store image and action only if unique (check only one channel)
				equal = True
				for i in range(len(training_img_set)):
					for row in range(len(training_img_set[i][0])):
						for col in range(len(training_img_set[i][0][row])):
							if(img[0][row][col] != training_img_set[i][0][row][col]):
								equal = False
								break
				if not equal:
					training_img_set.append(img)
					print("new")
			else:
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
gatherData(images_filename)



