import numpy as np
import random
import matplotlib.pyplot as plt
import network
import data_manager
import time

#Define Neural Network parameters
N = 1001
theta = 350
f = 0.5
gamma = 6

#Define Patterns to be learnt
learning_length = 50
repeat_sequence = 1
shuffle = False
nbr_patterns = 5
patterns = (np.random.rand(nbr_patterns, N) < f).astype(int)

#Define learning parameters
learn_rate = 0.01
eps = 1.2 #robustness

#Testing parameters
test_length = 100
b = 0.5 #basin size
#successful_storage_rate=0.9
#test_nb_iter=30
#ham_dist_threshold=0.01
		

results = data_manager.Data_manager('results.txt')
NN = 1

if False: #Do some training
	for eps in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10.]:
		#Initiliaze Neural Network
		del NN
		NN = network.Network(N, theta, f, gamma)		
		#Learn patterns
		NN.Three_TLR_training(patterns, learn_rate, eps, nb_iter=learning_length, repeat_sequence=repeat_sequence, shuffle=shuffle)
		#Test storage capacity as a function of basin size
		NN.testing(patterns, b, test_length)
		results.add_result(tuple(NN.id_options_results))
		
		
else: #no training, but we still need parameters to plot results
	NN = network.Network(N, theta, f, gamma)
	NN.set_params(patterns, learn_rate, eps, nb_iter=learning_length, repeat_sequence=repeat_sequence, shuffle=shuffle, b=b, test_length=test_length)
	


#Plot the hamming distance to patterns accross updates, without any excitations
NN.s = NN.add_noise_to_pattern(patterns[2], b)
NN.plot_convergence_to_patterns(patterns, nb_iter=10)

#Plot errors, hamming_distances, etc as a function of eps
id_options_results = NN.id_options_results[1]
#id_options_results['eps'] = 1.0 # If we need to change arguments for the plot
results.plot_results(id_options_results, x='eps', y=['err', 'avg_dist'], x_log_scale=True, figure=2, \
                     plot_points=True, pointargs=[['ok'], ['or']], pointkargs=[{'markersize':5}, {'markersize':3}],
                     plot_mean=True, meanargs=[['k'], ['r']], meankargs=[{'linewidth':2}, {'linewidth':1}])

plt.show()
