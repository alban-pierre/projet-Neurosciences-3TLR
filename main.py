import numpy as np
import math
import random

N = 100
theta = 0.2
f = 0.5
H0 = 0.2
H1 = 0.2
lmbda = 0.9
gamma = 6

nbr_patterns = 10
learning_length = 100
learn_rate = 0.1
epsilon = 1.2

execfile('network.py')

hamming_distance = lambda x, y: float(abs(x-y).sum())/x.size

NN = Network(N, theta, f, H0, H1, lmbda, gamma)

patterns = (np.random.rand(nbr_patterns, N) < f) + 0


#for ip in range(patterns.shape[0]):
for i in range(learning_length):
	ip = random.randint(0, nbr_patterns-1)
	NN.update_states(patterns[ip, :])
	NN.update_weights(learn_rate, epsilon)


test_length = 1
b = 0.1
err = 0
for i in range(test_length):
	ip = random.randint(0, nbr_patterns-1)
	p = np.copy(patterns[ip, :])
	r = range(N)
	random.shuffle(r)
	r = r[:int(round(nbr_patterns*N))]
	p[r] = (np.random.rand(round(nbr_patterns*N)) < f) + 0

	NN.s = p
	for j in range(30):
		NN.update_states()
	
	d = hamming_distance(patterns[ip, :], NN.s)
	err = err + (d > 0.01)


	
err = err/test_length

print "The percentage of recovery at size {} is {}%".format(b, 100-err*100)
