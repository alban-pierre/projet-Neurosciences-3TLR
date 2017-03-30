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
learning_length=100
nbr_patterns = 10
patterns = (np.random.rand(nbr_patterns, N) < f).astype(int)

#Define learning parameters
learn_rate = 0.01
eps = 1.2 #robustness

#Hamming distance
hamming_distance = lambda x, y: float(abs(x-y).sum())/x.size

#Initiliaze Neural Network
NN = network.Network(N, theta, f, gamma)

#Learn patterns
time1 = time.time()
for i_pattern in range(patterns.shape[0]):
	NN.Three_TLR(patterns[i_pattern], learn_rate, eps, nb_iter=learning_length)
print "The training of the neural network took {} seconds.".format(time.time() - time1)

#Test storage capacity as a function of basin size
test_length = 100
b = 0.5 #basin size
#successful_storage=np.zeros(patterns.shape[0]) #successful storage at basin size for the various patterns
err=np.zeros(patterns.shape[0]) #successful storage at basin size for the various patterns
successful_storage_rate=0.9

time2 = time.time()
for i_pattern in range(patterns.shape[0]):
	#err=0
	for i in range(test_length):
		pat = np.copy(patterns[i_pattern])
		random_part = range(N)
		random.shuffle(random_part)
		random_part = random_part[:int(round(N*b))]
		pat[random_part] = (np.random.rand(int(round(N*b))) < f).astype(int)
		NN.s = pat
		NN.update_states(np.zeros(N), nb_iter=30)
		d = hamming_distance(patterns[i_pattern], NN.s)
		err[i_pattern] = err[i_pattern] + int((d > 0.01))
	err[i_pattern] = err[i_pattern]/test_length
successful_storage=(1-err>successful_storage_rate).astype(int)
print "The testing of the neural network took {} seconds.".format(time.time() - time2)

print "The percentage of recovery is {}%.".format(100-err.mean()*100)
print "The percentage of patterns successfully stored is {}%.".format(100*successful_storage.mean())
