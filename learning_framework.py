#!/usr/bin/python
from __future__ import print_function
from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution

import numpy as np
import cv2
import itertools as it

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import model_from_json

# image parameters
downsampled_x = 64 #64
downsampled_y = 48#48
channels = 3 #channels on input image considered (GRAY8 = 1; RGB = 3)
skiprate = 5

class CustomDoomGame:
	def __init__(self,game,scenario, config,selectedMap = "map01"):
		game.set_vizdoom_path("../ViZDoom/bin/vizdoom")
		game.set_doom_game_path("../ViZDoom/scenarios/doom2.wad")
		game.set_doom_scenario_path(scenario)
		game.set_doom_map(selectedMap)
		game.load_config(config)

def start_game(game,multiplayer,visible,mode = Mode.PLAYER):
	if not visible:
		game.set_screen_resolution(ScreenResolution.RES_160X120)
	else:
		game.set_screen_resolution(ScreenResolution.RES_640X480)
	game.set_window_visible(visible)
	game.set_mode(mode)
	if multiplayer:
		# Start multiplayer game only with Your AI (with options that will be used in the competition, details in cig_host example).
		game.add_game_args("-host 1 -deathmatch +timelimit 10.0 "
			"+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
		# Name Your AI.
		game.add_game_args("+name AI")
		game.init()
	else:
		game.init()

# Function for converting images
def convert(img,colorCorrection=False,num_channels=0):
	if num_channels == 0:
		num_channels = channels
	
	if num_channels == 1:
		if not colorCorrection:
			img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		else:
			img = cv2.Canny(img,200,200)
			#cv2.imshow('Doom Buffer',img)
			#cv2.waitKey(1)
		img = cv2.resize(img, (downsampled_x, downsampled_y) )
		img_p = img.reshape([1,downsampled_y,downsampled_x])
	else:
		if colorCorrection:
			'''
			clip = (255-np.mean(img)) * 0.1#10
			clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(16,16))
			img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
			lab_plane = img_lab[:,:,0]
			cl2 = clahe.apply(lab_plane)
			img_lab[:,:,0] = cl2
			img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
			'''
			img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
			#cv2.imshow('Doom Buffer',img)
			#cv2.waitKey(1)
		# for RGB images
		img = img.astype(np.float32) / 255.0
		img_p = []
		for channel in range(channels):
			img_p.append(cv2.resize(img[:,:,channel], (downsampled_x, downsampled_y)))
	return np.array(img_p)


def create_cnn(input_rows,input_cols,num_outputs,final_activation='tanh'):
	model = Sequential()

	# input: input_colsxinput_rows images with 1 channels
	# this applies 32 convolution filters of size 3x3 each.
	'''
	model.add(Convolution2D(8, 3, 3,
                        border_mode='valid',
                        input_shape=(channels, input_rows, input_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) #32x24
	model.add(Convolution2D(8, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) #16x12
	model.add(Convolution2D(6, 2, 2))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2))) #8x6
	model.add(Flatten())
	model.add(Dense(num_outputs))
	model.add(Activation(final_activation)) # num_outputs
	
	model.compile(loss='categorical_crossentropy',
		optimizer='adadelta',
		metrics=['accuracy'])
	'''
	model.add(Convolution2D(24, 7, 7,
			border_mode='valid',
			input_shape=(channels, input_rows, input_cols)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(32, 4, 4))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(48, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dense(num_outputs))
	model.add(Activation(final_activation)) # num_outputs
	
	model.compile(loss='categorical_crossentropy',
		optimizer='adadelta',
		metrics=['accuracy'])
	
	return model


def get_available_actions(num_actions):
	# Creates all possible actions.
	actions = []
	for perm in it.product([0, 1], repeat=num_actions):
		actions.append(list(perm))
	return actions


def detectImage(screen):
    detected = False
    location = None
    img1 = cv2.imread('cacodemon.png',1)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb = cv2.ORB(1000,1.2)
    kp1, des1 = orb.detectAndCompute(img1,None)
	
    MIN_MATCH_COUNT = 90
    kp2, des2 = orb.detectAndCompute(screen,None)
    if not des2 is None:
        matches = bf.match(des1,des2)
    else:
        matches = []
	# Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    good = []
    points = []
    points2 = []
        
    if (len(matches) > MIN_MATCH_COUNT):
        for i in range (0,MIN_MATCH_COUNT):
            good.append(matches[i])
		
	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w, depth = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        avgX  = 0.0
        for b in dst_pts:
            avgX = avgX + b[0][0]
        avgX = avgX / dst_pts.size * 2
			
        avgY  = 0.0
        for b in dst_pts:
            avgY = avgY + b[0][1]
        avgY = avgY / dst_pts.size * 2

        #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        #img2 = cv2.line(img2, (0,0), (int(avgX), int(avgY)),255,3,cv2.LINE_AA)
        print("Matches Found-----------------------: - %d/%d" % (len(matches),MIN_MATCH_COUNT))
			
        #global detected
        detected = True
        #global location
        location = (avgX,avgY)

    else:
        print ("Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT))
        matchesMask = None
	
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = matchesMask, # draw only inliers
               flags = 2)
    return detected, location

        
