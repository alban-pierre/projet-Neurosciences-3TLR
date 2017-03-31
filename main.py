import numpy as np
import random
import network
import time

#Define Neural Network parameters
N = 1001
theta = 350
f = 0.5
gamma = 6

#Define Patterns to be learnt
learning_length=30
nbr_patterns = 5
patterns = (np.random.rand(nbr_patterns, N) < f).astype(int)

#Define learning parameters
learn_rate = 0.01
eps = 1.2 #robustness

#Initiliaze Neural Network
NN = network.Network(N, theta, f, gamma)

#Learn patterns
NN.Three_TLR_training(patterns, learn_rate, eps, nb_iter=learning_length, repeat_sequence=1, shuffle=False)

#Test storage capacity as a function of basin size
test_length = 100
b = 0.1 #basin size
NN.testing(patterns, b=0.1, test_length=100)


#Plot the hamming distance to patterns accross updates, without any excitations
NN.s = NN.add_noise_to_pattern(patterns[2])
NN.plot_convergence_to_patterns(patterns, nb_iter=10)
