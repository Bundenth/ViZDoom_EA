#!/usr/bin/env python

import itertools as it
import pickle
from random import sample, randint, random
from time import time
from vizdoom import *

import cv2
import numpy as np
import theano
from lasagne.init import GlorotUniform, Constant, HeNormal
from lasagne.layers import Conv2DLayer, InputLayer, DenseLayer, MaxPool2DLayer, get_output, get_all_params, \
    get_all_param_values, set_all_param_values
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
from theano import tensor
from tqdm import *
from time import sleep

from learning_framework import *
import learning_framework

controller_weights_filename = 'full_RL/pursuit_and_gather/controller_weights_0.save'
doom_scenario = "scenarios/pursuit_and_gather.wad"
doom_config = "config/pursuit_and_gather.cfg"
evaluation_filename = "full_RL/pursuit_and_gather/evaluation_0.txt"
stats_file = "full_RL/pursuit_and_gather/_stats_0.txt"

load_previous_net = True # use previously trained network to resume training
isCig = False
isTraining = False
useShapingReward = True
useShapingRewardInTesting = False
slowTestEpisode = True

#when using feature detector net
use_feature_detector = False
feature_weights_filename = 'feature_detector_nets/cig/FD_64x48x16_distanceL_0.save'
num_features = 16

if "health_gathering_supreme" in doom_scenario:
	#left, right,forward, backward and pair-combinations (health_gathering)
	actions_available = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1], #single actions
				[1,0,1,0], [0,1,1,0], [1,0,0,1], [0,1,0,1]] #let-right,forward,backward double actions
else:
	# left,right, forward and shoot and pair-combinations (pursuit and gather)
	actions_available = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1], #single actions
				[1,0,1,0],[0,1,1,0], #let-right,forward double actions
				[1,0,0,1],[0,1,0,1],[0,0,1,1]] #single actions+shoot
	#left,right,strafe both, forward, shoot
	#actions_available = [ [1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],
	#			[1,0,0,0,1,0],[0,0,0,0,1,0],
	#			[1,0,0,0,0,1],[0,1,0,0,0,1],[0,0,1,0,0,1],[0,0,0,1,0,1],[0,0,0,0,1,1]]

# Q-learning settings:
replay_memory_size = 20000
discount_factor = 0.99
start_epsilon = float(1.0)
end_epsilon = float(0.1)
epsilon = start_epsilon
static_epsilon_steps = 10000
epsilon_decay_steps = 100000
epsilon_decay_stride = (start_epsilon - end_epsilon) / epsilon_decay_steps

# Max reward is about 100 (for killing) so it'll be normalized
reward_scale = 0.01

# Some of the network's and learning settings:
learning_rate = 0.00001
batch_size = 48
epochs = 500
training_steps_per_epoch = 5000
test_episodes_per_epoch = 10

# Other parameters
evaluation_episodes = 100

#rewards
if "health_gathering_supreme" in doom_scenario:
	ammo_reward = 0.0
	shooting_reward = 0.0
	health_reward = 0.0
	harm_reward = 0.0
else:
	ammo_reward = 50.0
	shooting_reward = -30.0
	health_reward = 75.0
	harm_reward = -50.0
	
# shaping reward
initial_health = 100

# Replay memory:
class ReplayMemory:
    def __init__(self, capacity):

        if not use_feature_detector:
            state_shape = (capacity, channels, downsampled_y,downsampled_x)
        else:
            state_shape = (capacity, 1, num_features)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.nonterminal = np.zeros(capacity, dtype=np.bool_)

        self.size = 0
        self.capacity = capacity
        self.oldest_index = 0
        
        self.last_shaping_reward = 0
        self.last_ammo = -1
        self.last_health = initial_health

    def add_transition(self, s1, action, s2, reward):
        if not use_feature_detector:
            self.s1[self.oldest_index] = s1
        else:
            self.s1[self.oldest_index,0] = s1
       
        if s2 is None:
            self.nonterminal[self.oldest_index] = False
        else:
            if not use_feature_detector:
                self.s2[self.oldest_index] = s2
            else:
                self.s2[self.oldest_index,0] = s2
            self.nonterminal[self.oldest_index] = True
            
        self.a[self.oldest_index] = action
        self.r[self.oldest_index] = reward

        self.oldest_index = (self.oldest_index + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.s2[i], self.a[i], self.r[i], self.nonterminal[i]


# Creates the network:
def create_network(available_actions_num):
	# Creates the input variables
	if use_feature_detector:
		s1 = tensor.tensor3("States")
	else:
		s1 = tensor.tensor4("States")
	
	a = tensor.vector("Actions", dtype="int32")
	q2 = tensor.vector("Next State best Q-Value")
	r = tensor.vector("Rewards")
	nonterminal = tensor.vector("Nonterminal", dtype="int8")

	if use_feature_detector:
		# Creates the input layer of the network.
		dqn = InputLayer(shape=[None, 1, num_features], input_var=s1)

		dqn = DenseLayer(dqn, num_units=int(num_features * 0.5), nonlinearity=rectify, W=GlorotUniform("relu"),
						 b=Constant(.1))

		# Adds a single fully connected layer which is the output layer.
		# (no nonlinearity as it is for approximating an arbitrary real function)
		dqn = DenseLayer(dqn, num_units=available_actions_num, nonlinearity=None)
	else:
		# Creates the input layer of the network.
		dqn = InputLayer(shape=[None, channels, downsampled_y, downsampled_x], input_var=s1)

		# Adds 3 convolutional layers, each followed by a max pooling layer.
		dqn = Conv2DLayer(dqn, num_filters=32, filter_size=[7, 7],
						  nonlinearity=rectify, W=GlorotUniform("relu"),
						  b=Constant(.1))
		dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
		dqn = Conv2DLayer(dqn, num_filters=48, filter_size=[4, 4],
						  nonlinearity=rectify, W=GlorotUniform("relu"),
						  b=Constant(.1))

		dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
		dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[3, 3],
						  nonlinearity=rectify, W=GlorotUniform("relu"),
						  b=Constant(.1))
		dqn = MaxPool2DLayer(dqn, pool_size=[2, 2])
		# Adds a single fully connected layer.
		dqn = DenseLayer(dqn, num_units=128, nonlinearity=rectify, W=GlorotUniform("relu"),
						 b=Constant(.1))

		# Adds a single fully connected layer which is the output layer.
		# (no nonlinearity as it is for approximating an arbitrary real function)
		dqn = DenseLayer(dqn, num_units=available_actions_num, nonlinearity=None)

	# Theano stuff
	q = get_output(dqn)
	# Only q for the chosen actions is updated more or less according to following formula:
	# target Q(s,a,t) = r + gamma * max Q(s2,_,t+1)
	target_q = tensor.set_subtensor(q[tensor.arange(q.shape[0]), a], r + discount_factor * nonterminal * q2)
	loss = squared_error(q, target_q).mean()

	# Updates the parameters according to the computed gradient using rmsprop.
	params = get_all_params(dqn, trainable=True)
	updates = rmsprop(loss, params, learning_rate)

	# Compiles theano functions
	print "Compiling the network ..."
	function_learn = theano.function([s1, q2, a, r, nonterminal], loss, updates=updates, name="learn_fn")
	function_get_q_values = theano.function([s1], q, name="eval_fn")
	function_get_best_action = theano.function([s1], tensor.argmax(q), name="test_fn")
	print "Network compiled."

	# Returns Theano objects for the net and functions.
	# We wouldn't need the net anymore but it is nice to save your model.
	return dqn, function_learn, function_get_q_values, function_get_best_action


# Makes an action according to epsilon greedy policy and performs a single backpropagation on the network.
def perform_learning_step():
    # Checks the state and downsamples it.
    if not use_feature_detector:
        s1 = convert(game.get_state().image_buffer)
    else:
        img_p = convert(game.get_state().image_buffer).reshape([1, channels, downsampled_y, downsampled_x])
        s1 = (fd_network.predict(img_p).flatten()).reshape([1,1,num_features]).astype(np.float32)

    # With probability epsilon makes a random action.
    if random() <= epsilon:
        a = randint(0, len(actions_available) - 1)
    else:
        # Chooses the best action according to the network.
        if not use_feature_detector:
            a = get_best_action(s1.reshape([1, channels, downsampled_y, downsampled_x]))
        else:
            a = get_best_action(s1)
    
    reward = game.make_action(actions_available[a], skiprate + 1)
    
    #shaping rewards
    ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
    if not memory.last_ammo < 0:
        if ammo > memory.last_ammo:
            reward += ammo_reward
        if ammo < memory.last_ammo:
            reward += shooting_reward
    memory.last_ammo = ammo
    health = max(0,game.get_game_variable(GameVariable.HEALTH))
    if health > memory.last_health:
        reward += health_reward
    if health < memory.last_health - 5:
		reward += harm_reward
    memory.last_health = health
    sr = doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
    sr = sr - memory.last_shaping_reward
    memory.last_shaping_reward += sr
    reward += sr
    
    reward *= reward_scale

    if game.is_episode_finished():
        s2 = None
    else:
        if(not use_feature_detector):
            s2 = convert(game.get_state().image_buffer)
        else:
            img_p = convert(game.get_state().image_buffer).reshape([1, channels, downsampled_y, downsampled_x])
            s2 = np.reshape(fd_network.predict(img_p).flatten(),(1,num_features))

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, reward)

    # Gets a single, random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, s2, a, reward, nonterminal = memory.get_sample(batch_size)
        q2 = np.max(get_q_values(s2), axis=1)
        loss = learn(s1, q2, a, reward, nonterminal)
    else:
        loss = 0
    return loss


# Creates all possible actions.
#n = game.get_available_buttons_size()
#actions = []
#for perm in it.product([0, 1], repeat=n):
#    actions.append(list(perm))

# Creates replay memory which will store the transitions
memory = ReplayMemory(capacity=replay_memory_size)
net, learn, get_q_values, get_best_action = create_network(len(actions_available))

# Loads the  network's parameters if the loadfile was specified
if load_previous_net or not isTraining:
    params = pickle.load(open(controller_weights_filename, "r"))
    set_all_param_values(net, params)
    

if isTraining:
	# Creates and initializes the environment.
	print "Initializing doom..."
	game = DoomGame()
	CustomDoomGame(game,doom_scenario,doom_config)
	start_game(game,isCig,False)
	print "Doom initialized."
	if use_feature_detector:
		fd_network = create_cnn(downsampled_y,downsampled_x,num_features,'linear')
		fd_network.load_weights(feature_weights_filename)
	else:
		fd_network = None
	print "Starting the training!"
	
	f = open(stats_file,'w')
	f.write("train_mean,train_st_dev,train_max,train_min,train_loss,train_epsilon,test_mean,test_st_dev,test_max,test_min\n")
	f.close()

	steps = 0
	for epoch in range(epochs):
		print "\nEpoch", epoch
		train_time = 0
		train_episodes_finished = 0
		train_loss = []
		train_rewards = []

		train_start = time()
		print "\nTraining ..."
		game.new_episode()
		for learning_step in tqdm(range(training_steps_per_epoch)):
			# Learning and action is here.
			train_loss.append(perform_learning_step())
			# I
			if game.is_episode_finished():
				r = game.get_total_reward()
				train_rewards.append(r)
				game.new_episode()
				memory.last_shaping_reward = 0
				memory.last_ammo = -1
				memory.last_health = initial_health
				train_episodes_finished += 1

			steps += 1
			if steps > static_epsilon_steps:
				epsilon = max(end_epsilon, epsilon - epsilon_decay_stride)

		train_end = time()
		train_time = train_end - train_start
		mean_loss = np.mean(train_loss)

		print train_episodes_finished, "training episodes played."
		print "Training results:"

		train_rewards = np.array(train_rewards)

		print "mean:", train_rewards.mean(), "std:", train_rewards.std(), "max:", train_rewards.max(), "min:", train_rewards.min(), "mean_loss:", mean_loss, "epsilon:", epsilon
		print "t:", str(round(train_time, 2)) + "s"
		
		#store training stats
		f = open(stats_file,'a')
		f.write(str(train_rewards.mean()) + ',' + str(train_rewards.std()) + ',' + str(train_rewards.max()) + ',' + str(train_rewards.min()) + ',' + str(mean_loss) + ',' + str(epsilon) + ',')
		f.close()
		
		# Testing
		test_episode = []
		test_rewards = []
		test_start = time()
		print "Testing..."
		for test_episode in tqdm(range(test_episodes_per_epoch)):
			game.new_episode()
			while not game.is_episode_finished():
				state = convert(game.get_state().image_buffer).reshape([1, channels, downsampled_y, downsampled_x])
				if not use_feature_detector:
					best_action_index = get_best_action(state)
				else:
					s1 = (fd_network.predict(state).flatten()).reshape([1,1,num_features]).astype(np.float32)
					best_action_index = get_best_action(s1)
					
					
				game.make_action(actions_available[best_action_index], skiprate + 1)
				if game.is_player_dead():
					break
			r = game.get_total_reward()
			test_rewards.append(r)

		test_end = time()
		test_time = test_end - test_start
		print "Test results:"
		test_rewards = np.array(test_rewards)
		print "mean:", test_rewards.mean(), "std:", test_rewards.std(), "max:", test_rewards.max(), "min:", test_rewards.min()
		print "t:", str(round(test_time, 2)) + "s"
		
		#store testing stats
		f = open(stats_file,'a')
		f.write(str(test_rewards.mean()) + ',' + str(test_rewards.std()) + ',' + str(test_rewards.max()) + ',' + str(test_rewards.min()) + str("\n"))
		f.close()
		
		if controller_weights_filename:
			print "Saving network weigths to:", controller_weights_filename
			pickle.dump(get_all_param_values(net), open(controller_weights_filename, "w"))
		print "========================="
	game.close()

print "Time to watch!"

game = DoomGame()
CustomDoomGame(game,doom_scenario,doom_config)
start_game(game,isCig,slowTestEpisode)

if use_feature_detector:
	fd_network = create_cnn(downsampled_y,downsampled_x,num_features,'linear')
	fd_network.load_weights(feature_weights_filename)
else:
	fd_network = None

# Sleeping time between episodes, for convenience.
if slowTestEpisode:
	episode_sleep = 0.028
else:
	episode_sleep = 0.0
	f = open(evaluation_filename,'w')
	f.write('total reward' + str("\n"))
	f.close()


for i in range(evaluation_episodes):
	game.new_episode()
	ammo_reward = 0
	reward = 0
	last_ammo = -1
    
	while not game.is_episode_finished():
		state = convert(game.get_state().image_buffer).reshape([1, channels, downsampled_y, downsampled_x])
		if not use_feature_detector:
			best_action_index = get_best_action(state)
		else:
			s1 = (fd_network.predict(state).flatten()).reshape([1,1,num_features]).astype(np.float32)
			best_action_index = get_best_action(s1)
		ammo = game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
		game.set_action(actions_available[best_action_index])
		for i in range(0+1):
			game.advance_action()
		sleep(episode_sleep)
		if not last_ammo < 0:
			if ammo < last_ammo:
				ammo_reward += shooting_reward
		last_ammo = ammo
		if game.is_player_dead():
			break

	reward = game.get_total_reward()
	if useShapingRewardInTesting:
		reward += ammo_reward
	print "Reward: ", reward
	if not slowTestEpisode:
		#store stats
		f = open(evaluation_filename,'a')
		f.write(str(reward) + str("\n"))
		f.close()

game.close()
