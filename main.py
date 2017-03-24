import numpy as np
import math

N = 100
theta = 0.2
f = 0.5
H0 = 0.2
H1 = 0.2
lmbda = 0.9
gamma = 0.5

nbr_patterns = 10

NN = Network(N, theta, f, H0, H1, lmbda, gamma)

patterns = (np.random.rand((nbr_patterns, N)) < f) + 0



