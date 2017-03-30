import numpy as np
import random
import network

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
for i_pattern in range(patterns.shape[0]):
	NN.Three_TLR(patterns[i_pattern], learn_rate, eps, nb_iter=learning_length)


#Test storage capacity as a function of basin size
test_length = 100
b = 0.1 #basin size
successful_storage=np.zeros(patterns.shape[0]) #successful storage at basin size for the various patterns
successful_storage_rate=0.9
for i_pattern in range(patterns.shape[0]):
    err=0
    for i in range(test_length):
        pat = np.copy(patterns[i_pattern])
        random_part = range(N)
        random.shuffle(random_part)
        random_part = random_part[:int(round(N*b))]
        pat[random_part] = (np.random.rand(int(round(N*b))) < f).astype(int)
        NN.s = pat
        NN.update_states(np.zeros(N), nb_iter=30)
        d = hamming_distance(patterns[i_pattern], NN.s)
        err = err + int((d > 0.01))
    err = err/test_length
    successful_storage[i_pattern]=int((1-err>successful_storage_rate))

print "The percentage of recovery at size {} is {}%".format(b, 100-err*100)
