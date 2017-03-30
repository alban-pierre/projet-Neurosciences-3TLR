import numpy as np
from scipy.special import erfcinv
import math

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
		self.H0 = (self.N-1)*(self.f*self.W[self.mask].mean() - self.theta/(self.N-1)) + self.W[self.mask].std()*math.sqrt(2)*erfcinv(2*f)*math.sqrt((self.N-1)*f)
		self.H1 = 0					       
		self.inhibition= self.H0 + self.lmbda*(self.s.sum() - self.f*self.N)
		self.v = np.dot(self.W,self.s)-self.inhibition
		
	def update_states(self, external_input, nb_iter=1): #update neurons states
		self.lmbda = self.W[self.mask].mean()
		self.H0 = (self.N-1)*(self.f*self.W[self.mask].mean() - self.theta/(self.N-1)) + self.W[self.mask].std()*math.sqrt(2)*erfcinv(2*self.f)*math.sqrt((self.N-1)*self.f)
		self.H1 = self.f*self.gamma*math.sqrt(self.N-1) 
		for k in range(nb_iter):
			self.inhibition = self.H0 + self.H1*external_input.sum()/(self.f*self.N) + self.lmbda*(self.s.sum() - self.f*self.N)
			self.v = np.dot(self.W, self.s) + self.gamma*math.sqrt(self.N)*external_input - self.inhibition
			self.s = self.activation_func(self.v)

	def Three_TLR(self, pattern, learn_rate, eps, nb_iter=100): #learning pattern
		self.update_states(pattern) #update states to set the network into the presented pattern
	 #Define learning thresholds
		theta0 = self.theta - (self.gamma + eps)*self.f*math.sqrt(self.N)
		theta1 = self.theta + (self.gamma + eps)*self.f*math.sqrt(self.N)
	 #Update weights
		for k in range(nb_iter):	
			coeffs = - (self.v < self.theta).astype(int) + (self.v <= theta0).astype(int) + (self.v > self.theta).astype(int) - (self.v >= theta1).astype(int)
			self.W = self.W + learn_rate*np.dot(np.reshape(coeffs,(coeffs.size,1)),np.reshape(self.s,(1,self.s.size)))
			np.fill_diagonal(self.W, 0)
			self.W = np.maximum(self.W, 0)
			self.update_states(pattern)
