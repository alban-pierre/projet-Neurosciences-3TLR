# Main function tu run, it loads a network, train it and stores the result on the disk. It also plots results computed and saved results

import numpy as np
import random
import matplotlib.pyplot as plt
import network
import data_manager
import time

# +----------------------------------------+
# | Beginning of neural network parameters |
# +----------------------------------------+

N = 201 # Number of neurons of the network
theta = 70 # Threshold of the actiation function 
f = 0.5 # Proportion of ones in the patterns and in the network states
gamma = 6

# Define Patterns to be learnt
learning_length = 10 # Number of sequencial updates for each pattern presented
repeat_sequence = 100 # Number of times a pattern is presented
shuffle = False # If we randomize the order of presentation of patterns
nbr_patterns = 50 # Number of patterns we want to learn

# Define learning parameters
learn_rate = 0.003 # Learning rate
eps = 1.8 # Robustness

# Testing parameters
test_length = 100 # Number of tests for each pattern
b = 0.1 # Basin size
# successful_storage_rate = 0.9 # Minimum proportion to say 'this pattern is successfully stored'
# test_nb_iter = 30 # After how many iteration we compute the hamming distance between states of the network and the pattern we want to retrieve
# ham_dist_threshold = 0.01 # Error we accept in the hamming distance

plot_histograms=False # If we want to plot histograms of v
do_training=True # If we want to make some networks and train them

# +----------------------------------+
# | End of neural network parameters |
# +----------------------------------+

results = data_manager.Data_manager('results2.txt')
NN = 1

if plot_histograms:
	NN = network.Network(N, theta, f, gamma)
	NN.plot_v(np.zeros(N), eps, figure=3, color='blue')
	plt.xlabel("v")
	plt.ylabel("v occurrences")
	plt.title("v occurences before learning without external input")
	NN.plot_v((np.random.rand(1, N) < f).astype(int), eps, figure=4, color='blue')
	plt.xlabel("v")
	plt.ylabel("v occurrences")
	plt.title("v occurences before learning with external input")
	
if do_training: #Do some training
	#for gamma in [6]:
	#for eps in [2.4, 2.7, 3.0]:#[1.2, 1.5, 1.8, 2.1]:
	for nbr_patterns in [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100]:
		patterns = (np.random.rand(nbr_patterns, N) < f).astype(int)
		#Initiliaze Neural Network
		del NN
		NN = network.Network(N, theta, f, gamma)		
		#Learn patterns
		NN.Three_TLR_training(patterns, learn_rate=learn_rate, eps=eps, nb_iter=learning_length, repeat_sequence=repeat_sequence, shuffle=shuffle)
		#Test storage capacity as a function of basin size
		NN.testing(patterns, b=b, test_length=test_length)
		results.add_result(tuple(NN.id_options_results))
		
		
	#Plot the hamming distance to patterns accross updates, without any excitations
	NN.s = NN.add_noise_to_pattern(patterns[2], b)
	NN.plot_convergence_to_patterns(patterns, nb_iter=10)
	
	
	
else: #no training, but we still need parameters to plot results
	patterns = (np.random.rand(nbr_patterns, N) < f).astype(int)
	NN = network.Network(N, theta, f, gamma)
	NN.set_params(patterns, learn_rate, eps, nb_iter=learning_length, repeat_sequence=repeat_sequence, shuffle=shuffle, b=b, test_length=test_length)

if plot_histograms:
	NN.s = patterns[4]
	NN.plot_v(np.zeros(N), eps, figure=5, color='blue')
	plt.xlabel("v")
	plt.ylabel("v occurrences")
	plt.title("v occurences after learning without external input, last pattern")
	NN.s = patterns[4]
	NN.plot_v(patterns[4], eps, figure=6, color='blue')
	plt.xlabel("v")
	plt.ylabel("v occurrences")
	plt.title("v occurences after learning with external input, last pattern")
	NN.s = patterns[2]
	NN.plot_v(np.zeros(N), eps, figure=7, color='blue')
	plt.xlabel("v")
	plt.ylabel("v occurrences")
	plt.title("v occurences after learning without external input, random pattern")
	NN.s = patterns[2]
	NN.plot_v(patterns[2], eps, figure=8, color='blue')
	plt.xlabel("v")
	plt.ylabel("v occurrences")
	plt.title("v occurences after learning with external input, random pattern")


#Plot errors, hamming_distances, etc as a function of eps
id_options_results = NN.id_options_results[1].copy()
#del id_options_results['eps']
#id_options_results['eps'] = 3.0
#results.plot_results(id_options_results, x='nbr_patterns', y=['err', 'successful_storage', 'avg_dist'], x_log_scale=False, figure=2, \
#                     plot_points=True, pointargs=[['ok'], ['ob'], ['or']], pointkargs=[{'markersize':3}, {'markersize':3}, {'markersize':3}],
#                     plot_mean=True, meanargs=[['k'], ['b'], ['r']], meankargs=[{'linewidth':2}, {'linewidth':2}, {'linewidth':2}],
#                     plot_std=False, stdargs=[['k'], ['b'], ['r']], stdkargs=[{'linewidth':1}, {'linewidth':1}, {'linewidth':1}],
#                     plot_minmax=True, minmaxargs=[['k--'], ['b--'], ['r--']], minmaxkargs=[{'linewidth':1}, {'linewidth':1}, {'linewidth':1}])
results.plot_results(id_options_results, x='nbr_patterns', y=['successful_storage'], x_log_scale=False, figure=2, \
                     plot_points=True, pointargs=[['ob']], pointkargs=[{'markersize':3}],
                     plot_mean=True, meanargs=[['b']], meankargs=[{'linewidth':2}],
                     plot_std=False, stdargs=[['b']], stdkargs=[{'linewidth':1}],
                     plot_minmax=False, minmaxargs=[['b--']], minmaxkargs=[{'linewidth':1}])
plt.ylabel("Successful storage proportion")
plt.title("Successful storage proportion as a function of #patterns")


plt.show()
