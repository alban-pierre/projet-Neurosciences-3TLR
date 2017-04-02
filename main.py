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
b = 0.1 #basin size
#successful_storage_rate=0.9
#test_nb_iter=30
#ham_dist_threshold=0.01
		

results = data_manager.Data_manager('results.txt')
NN = 1

plot_histograms=True

if plot_histograms:
	NN = network.Network(N, theta, f, gamma)
	NN.plot_v(np.zeros(N), eps, figure=3, color='blue')
	NN.plot_v((np.random.rand(1, N) < f).astype(int), eps, figure=4, color='green')

if True: #Do some training
	#for gamma in [3, 6, 12]:
		#for eps in [0.2, 0.5, 1., 2., 5.]:
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

if plot_histograms:
	NN.s = patterns[4]
	NN.plot_v(np.zeros(N), eps, figure=5, color='black')
	NN.s = patterns[4]
	NN.plot_v(patterns[4], eps, figure=6, color='red')
	NN.s = patterns[2]
	NN.plot_v(np.zeros(N), eps, figure=7, color='black')
	NN.s = patterns[2]
	NN.plot_v(patterns[2], eps, figure=8, color='red')


#Plot the hamming distance to patterns accross updates, without any excitations
NN.s = NN.add_noise_to_pattern(patterns[2], b)
NN.plot_convergence_to_patterns(patterns, nb_iter=10)

#Plot errors, hamming_distances, etc as a function of eps
id_options_results = NN.id_options_results[1]
#id_options_results['eps'] = 1.0 # If we need to change arguments for the plot
results.plot_results(id_options_results, x='eps', y=['err', 'successful_storage', 'avg_dist'], x_log_scale=True, figure=2, \
                     plot_points=True, pointargs=[['ok'], ['ob'], ['or']], pointkargs=[{'markersize':3}, {'markersize':3}, {'markersize':3}],
                     plot_mean=True, meanargs=[['k'], ['b'], ['r']], meankargs=[{'linewidth':2}, {'linewidth':2}, {'linewidth':2}],
                     plot_std=False, stdargs=[['k'], ['b'], ['r']], stdkargs=[{'linewidth':1}, {'linewidth':1}, {'linewidth':1}],
                     plot_minmax=False, minmaxargs=[['k--'], ['b--'], ['r--']], minmaxkargs=[{'linewidth':1}, {'linewidth':1}, {'linewidth':1}])



plt.show()
