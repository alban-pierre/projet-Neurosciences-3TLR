import numpy as np
from scipy.special import erfcinv
import matplotlib.pyplot as plt
import math
import time
import random

class Network:

	#Initialize network and define binary neural network attributes for the 3TLR 
	def __init__(self, N=1001, theta=350, f=0.5, gamma=6, initial_mean=1, initial_variance=1):
		self.N = N #Number of neurons in the network (binary Mac Culloch Pitts model)
		self.theta = float(theta) #Threshold in the step function to output post synaptic activity
		self.f = float(f) #Sparsity of the network (i.e percentage of active neurons)
		self.gamma = float(gamma) #Measures the strength of the external input to separate ON and OFF neurons in the first stage of the learning
		#Weight matrix - initialized as Gaussian with 1 mean and 1 variance, 0 diagonal weights (no self-connection), then floored to zero (positive weights to respect Dale's principle)
		self.W = initial_variance*np.random.randn(self.N, self.N) + initial_mean 
		np.fill_diagonal(self.W, 0)
		self.W = np.maximum(self.W, 0)
		self.activation_func = lambda x: (x > self.theta).astype(int) #Step activation function
		self.s = (np.random.random(self.N) < self.f).astype(int) #State of the network - initialized as random {0,1} states with f sparseness
		self.mask = np.ones(self.W.shape, dtype=bool) #mask to compute mean and standard deviation of the weight matrix (ignoring diagonal which are non random since set to 0)
		np.fill_diagonal(self.mask, 0)
		self.lmbda = self.W[self.mask].mean()
		self.H0 = (self.N-1)*(self.f*self.W[self.mask].mean() - self.theta/(self.N-1)) + self.W[self.mask].std()*math.sqrt(2)*erfcinv(2*self.f)*math.sqrt((self.N-1)*self.f)
		self.H1 = 0					       
		self.inhibition= self.H0 + self.lmbda*(self.s.sum() - self.f*self.N)
		self.v = np.dot(self.W,self.s)-self.inhibition
		self.id_options_results = [0, {'N':self.N, 'theta':self.theta, 'f':self.f, 'gamma':self.gamma, 'initial_mean':initial_mean, 'initial_variance':initial_variance}, {}]

		
	def update_states(self, external_input, nb_iter=1): #update neurons states
		self.lmbda = self.W[self.mask].mean()
		self.H0 = (self.N-1)*(self.f*self.W[self.mask].mean() - self.theta/(self.N-1)) + self.W[self.mask].std()*math.sqrt(2)*erfcinv(2*self.f)*math.sqrt((self.N-1)*self.f)
		self.H1 = self.f*self.gamma*math.sqrt(self.N-1)
		for k in range(nb_iter):
			self.inhibition = self.H0 + self.H1*external_input.sum()/(self.f*self.N) + self.lmbda*(self.s.sum() - self.f*self.N)
			self.v = np.dot(self.W, self.s) + self.gamma*math.sqrt(self.N)*external_input - self.inhibition
			self.s = self.activation_func(self.v)

			
	def Three_TLR(self, pattern, learn_rate, eps, nb_iter=100): #learning pattern
		#self.s = (np.random.random(self.N) < self.f).astype(int) #State of the network - initialized as random {0,1} states with f sparseness
		self.update_states(pattern) #update states to set the network into the presented pattern
		#Define learning thresholds
		#print "Hamming distance to pattern : {}".format(self.hamming_distance(pattern))
		theta0 = self.theta - (self.gamma + eps)*self.f*math.sqrt(self.N)
		theta1 = self.theta + (self.gamma + eps)*self.f*math.sqrt(self.N)
		#Update weights
		for k in range(nb_iter):	
			coeffs = - (self.v < self.theta).astype(int) + (self.v <= theta0).astype(int) + (self.v > self.theta).astype(int) - (self.v >= theta1).astype(int)
			#print "{}   :	 {}   /	  {}   ---   {}	  vs   {}".format(abs(coeffs).sum(), coeffs.sum(), self.hamming_distance(pattern), abs(np.dot(np.reshape(coeffs,(coeffs.size,1)),np.reshape(self.s,(1,self.s.size)))).sum(), (np.dot(np.reshape(coeffs,(coeffs.size,1)),np.reshape(self.s,(1,self.s.size)))).sum())
			self.W = self.W + learn_rate*np.dot(np.reshape(coeffs,(coeffs.size,1)),np.reshape(self.s,(1,self.s.size)))
			np.fill_diagonal(self.W, 0)
			self.W = np.maximum(self.W, 0)
			self.update_states(pattern)

			
	def Three_TLR_training(self, patterns, learn_rate=0.01, eps=1.2, nb_iter=100, repeat_sequence=1, shuffle=True):
		time1 = time.time()
		for i_seq in range(repeat_sequence):
			r = range(patterns.shape[0])
			if shuffle:
				random.shuffle(r)
			for i_pattern in r:
				self.Three_TLR(patterns[i_pattern], learn_rate, eps, nb_iter=nb_iter)
		print "The training of the neural network took {} seconds.".format(time.time() - time1)
		self.id_options_results[1].update({'nbr_patterns':patterns.shape[0], 'learn_rate':learn_rate, 'eps':eps, 'nb_iter':nb_iter, 'reapeat_sequence':repeat_sequence, 'shuffle':shuffle})

		
	def testing(self, patterns, b=0.1, test_length=100, successful_storage_rate=0.9, test_nb_iter=30, ham_dist_threshold=0.01):
		time2 = time.time()
		err=np.zeros(patterns.shape[0]) #successful storage at basin size for the various patterns
		all_d=np.zeros(patterns.shape[0]) #successful storage at basin size for the various patterns
		for i_pattern in range(patterns.shape[0]):
			for i in range(test_length):
				self.s = self.add_noise_to_pattern(patterns[i_pattern])
				self.update_states(np.zeros(self.N), nb_iter=test_nb_iter)
				d = self.hamming_distance(patterns[i_pattern])
				err[i_pattern] = err[i_pattern] + int((d > ham_dist_threshold))
				all_d[i_pattern] = all_d[i_pattern] + d
			err[i_pattern] = err[i_pattern]/test_length
			all_d[i_pattern] = all_d[i_pattern]/test_length
		successful_storage=(1-err>successful_storage_rate).astype(int)
		print "The testing of the neural network took {} seconds.".format(time.time() - time2)
		print "The average hamming distance between patterns and neurons states is {}.".format(all_d.mean())
		print "The percentage of recovery is {}%.".format(100-err.mean()*100)
		print "The percentage of patterns successfully stored is {}%.".format(100*successful_storage.mean())
		self.id_options_results[0] = int(time.time())
		self.id_options_results[1].update({'b':b, 'test_length':test_length, 'successful_storage_rate':successful_storage_rate, 'test_nb_iter':test_nb_iter, 'ham_dist_threshold':ham_dist_threshold})
		self.id_options_results[2].update({'err':1-err.mean(), 'successful_storage':successful_storage.mean(), 'avg_dist':all_d.mean()})
		return err


	def add_noise_to_pattern(self, pattern, b=0.1):
		pat = np.copy(pattern)
		random_part = range(self.N)
		random.shuffle(random_part)
		random_part = random_part[:int(round(self.N*b))]
		pat[random_part] = (np.random.rand(int(round(self.N*b))) < self.f).astype(int)
		return pat
		
	def hamming_distance(self, patterns): #hamming distance from patterns to the current state
		if (len(patterns.shape) == 1):
			return float(abs(patterns-self.s).sum())/self.N
		else:
			return abs(patterns-self.s).sum(axis=1).astype(float)/self.N
		
		
	def plot_convergence_to_patterns(self, patterns, nb_iter=10):
		d = np.zeros((nb_iter, patterns.shape[0]))
		d[0] = self.hamming_distance(patterns)
		for i in range(1,nb_iter):
			self.update_states(np.zeros(self.N))
			d[i] = self.hamming_distance(patterns)
		plt.plot(d)
		plt.xlabel('Number of updates')
		plt.ylabel('Hamming distance to patterns')
		plt.title('Hamming distance to patterns, no excitations')
		plt.show()

	
	def set_params(self, patterns, learn_rate=0.01, eps=1.2, nb_iter=100, repeat_sequence=1, shuffle=True, b=0.1, test_length=100, successful_storage_rate=0.9, test_nb_iter=30, ham_dist_threshold=0.01):
		self.id_options_results[1].update({'nbr_patterns':patterns.shape[0], 'learn_rate':learn_rate, 'eps':eps, 'nb_iter':nb_iter, 'reapeat_sequence':repeat_sequence, 'shuffle':shuffle})
		self.id_options_results[1].update({'b':b, 'test_length':test_length, 'successful_storage_rate':successful_storage_rate, 'test_nb_iter':test_nb_iter, 'ham_dist_threshold':ham_dist_threshold})
		
