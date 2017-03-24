import numpy as np
import math

class Network:

	def __init__(self, N=100, initial_mean=1, initial_variance=1):
		self.N = N
		self.W = initial_variance*numpy.random.randn(self.N, self.N) + initial_mean
		numpy.fill_diagonal(self.W, 0)
		self.W = numpy.maximum(self.W, 0)
		self.theta = 0.2
		self.activation_func = lambda x: (x > self.theta) + 0 # Later we can generalise the activation function as an argument given by the user
		self.s = (np.random.random(self.N) > 0.5) + 0 # Is there any casting method better than "+0" ?
		self.f = 0.5
		self.H0 = 0.2
		self.H1 = 0.2
		self.lmbda = 0.9
		self.gamma = 0.5
		self.vi = np.ones(self.N)*self.theta

	def update_states(self, xi):
		inhibition = self.H0 + self.H1*xi.sum()/(self.f*self.N*self.gamma*math.sqrt(self.N)) + self.lmbda*(self.s.sum() - self.f*self.N)
		self.vi = np.dot(self.W, self.s) + xi - inhibition
		self.s = self.activation_func(self.vi)

	def update_weights(self, learn_rate, epsilon):
		theta0 = self.theta - (self.gamma + epsilon)*self.f*math.sqrt(self.N)
		theta1 = self.theta + (self.gamma + epsilon)*self.f*math.sqrt(self.N)
		coeffs = - (self.vi < self.theta) + (self.vi > self.theta0) + (self.vi > self.theta) - (self.vi < self.theta1)
		self.W = self.W + coeffs*learn_rate*self.s


	
