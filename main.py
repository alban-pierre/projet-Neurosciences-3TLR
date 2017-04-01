import numpy as np
import random
import network
import data_manager
import time

#Define Neural Network parameters
N = 1001
theta = 350
f = 0.5
gamma = 4

#Define Patterns to be learnt
learning_length = 50
repeat_sequence = 1
shuffle = False
nbr_patterns = 5
patterns = (np.random.rand(nbr_patterns, N) < f).astype(int)

#Define learning parameters
learn_rate = 0.01
eps = 1.2 #robustness

results = data_manager.Data_manager('results.txt')
NN = 1

for eps in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10.]:
	#Initiliaze Neural Network
	del NN
	NN = network.Network(N, theta, f, gamma)
	
	#Learn patterns
	NN.Three_TLR_training(patterns, learn_rate, eps, nb_iter=learning_length, repeat_sequence=repeat_sequence, shuffle=shuffle)
	
	#Test storage capacity as a function of basin size
	test_length = 100
	b = 0.5 #basin size
	#successful_storage_rate=0.9
	#test_nb_iter=30
	#ham_dist_threshold=0.01
	NN.testing(patterns, b, test_length)
	
	results.add_result(tuple(NN.id_options_results))


#Plot the hamming distance to patterns accross updates, without any excitations
NN.s = NN.add_noise_to_pattern(patterns[2], b)
NN.plot_convergence_to_patterns(patterns, nb_iter=10)

#Plot errors, hamming_distances, etc as a function of eps
id_options_results = NN.id_options_results[1]
#id_options_results['eps'] = 1.0 # If we need to change arguments for the plot
results.plot_results(id_options_results, 'eps', ['err', 'avg_dist'])
