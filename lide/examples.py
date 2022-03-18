from .basis import *
import numpy as np 
from itertools import product
from polyrat import total_degree_index

def simple_harmonic_oscillator(total_degree = 1, alpha = 1, include_constant = True):
	r"""
	Parameters
	----------
	alpha: float
		Scale parameter for SHO (i.e., k/m)
	"""
	dim = 2
	assert total_degree >= 1
	index = total_degree_index(dim, total_degree)
	if include_constant is False:
		index = index[1:]

	C = np.zeros((dim, len(index)), dtype = float)

	phis = []
	for k, degree in enumerate(index):
		phis.append(Monomial(degree))
		if np.array_equal(degree, [0,1]):
			C[0,k] = 1 # x_1' = x_2
		if np.array_equal(degree, [1,0]):
			C[1,k] = -alpha

	x0 = np.array([1,0])
	return phis, C, x0

def damped_harmonic_oscillator(total_degree = 1, stiff = 2, damp = 0.1, include_constant = True):
	r"""
	From BKP16,
	Parameters
	----------
	alpha: float
		Scale parameter for SHO (i.e., k/m)
	"""
	dim = 2
	assert total_degree >= 1
	index = total_degree_index(dim, total_degree)
	if include_constant is False:
		index = index[1:]

	C = np.zeros((dim, len(index)), dtype = float)

	phis = []
	for k, degree in enumerate(index):
		phis.append(Monomial(degree))
		if np.array_equal(degree, [0,1]):
			C[0,k] = stiff # x_1' = x_2
			C[1,k] = -damp
		if np.array_equal(degree, [1,0]):
			C[1,k] = -stiff
			C[0,k] = -damp

	x0 = np.array([2,0])
	return phis, C, x0



def duffing(total_degree = 3, gamma = 0.1, kappa = 1, epsilon = 5):
	dim = 2
	assert total_degree >= 3, "Degree must be at least 3" 
	phis = []
	index = total_degree_index(dim, total_degree)
	C = np.zeros((dim, len(index)), dtype = float)
	

	for k, degree in enumerate(index):
		phis.append(Monomial(degree))
		if np.array_equal(degree, [0,1]):
			C[0,k] = 1         # x' = y
		if np.array_equal(degree, [0,1]):
			C[1,k] = -gamma    # y' += -gamma * y
		if np.array_equal(degree, [1,0]):
			C[1,k] = -kappa    # y' += -kappa*x 
		if np.array_equal(degree, [3,0]):
			C[1,k] = -epsilon  # y' += - epsilon * x**3

	x0 = np.array([-2,-2])
	return phis, C, x0


def lorenz63(total_degree = 2, sigma = 10, beta = 8/3, rho = 28):
	r"""
	Following notation on the wikipedia

	https://en.wikipedia.org/wiki/Lorenz_system

	"""
	assert total_degree >= 2
	phis = []
	dim = 3
	index = total_degree_index(dim, total_degree)
	C = np.zeros((dim, len(index)), dtype = float)
	
	for k, degree in enumerate(index):
		phis.append(Monomial(degree))

		if np.array_equal(degree, [1,0,0]):
			C[0,k] = -sigma
		if np.array_equal(degree, [0,1,0]):
			C[0,k] = sigma

		if np.array_equal(degree, [1,0,0]):
			C[1,k] = rho
		if np.array_equal(degree, [1,0,1]):
			C[1,k] = -1
		if np.array_equal(degree, [0,1,0]):
			C[1,k] = -1

		if np.array_equal(degree, [1,1,0]):
			C[2,k] = 1
		if np.array_equal(degree, [0,0,1]):
			C[2,k] = -beta
		
	x0 = np.array([-8,7,27])
	return phis, C, x0

def vanderpol(total_degree = 3, mu = 2):
	r"""

	https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
	"""
	assert total_degree >= 3
	phis = []
	dim = 2
	index = total_degree_index(dim, total_degree)
	C = np.zeros((dim, len(index)), dtype = float)
	
	for k, degree in enumerate(index):
		phis.append(Monomial(degree))
		if np.array_equal(degree, [0,1]):
			C[0,k] = 1

		if np.array_equal(degree, [0,1]):
			C[1,k] = mu
		if np.array_equal(degree, [2,1]):
			C[1,k] = -mu
		if np.array_equal(degree, [1,0]):
			C[1,k] = -1
		
	x0 = np.array([0,1])
	return phis, C, x0

def rossler(total_degree = 2, a = 0.2, b = 0.2, c = 5.7):
	r"""
	KBK20x
	"""
	assert total_degree >= 2
	phis = []
	dim = 3
	index = total_degree_index(dim, total_degree)
	C = np.zeros((dim, len(index)), dtype = float)
	
	for k, degree in enumerate(index):
		phis.append(Monomial(degree))
		if np.array_equal(degree, [0,1,0]):
			C[0,k] = -1
		if np.array_equal(degree, [0,0,1]):
			C[0,k] = -1
		if np.array_equal(degree, [1,0,0]):
			C[1,k] = 1
		if np.array_equal(degree, [0,1,0]):
			C[1,k] = a
		if np.array_equal(degree, [0,0,0]):
			C[2,k] = b
		if np.array_equal(degree, [1,0,1]):
			C[2,k] = 1
		if np.array_equal(degree, [0,0,1]):
			C[2,k] = -c

	x0 = np.array([3,5,0])

	return phis, C, x0
		
def lorenz96(total_degree = 3, F = 8, N = 4):
	assert total_degree >= 3
	phis = []
	dim = N + 2
	index = total_degree_index(dim, total_degree)
	C = np.zeros((dim, len(index)), dtype = float)
	
	dt = np.zeros(dim, dtype = np.int)
	for k, degree in enumerate(index):
		phis.append(Monomial(degree))

		# The constant term
		if np.sum(degree) == 0:
			for i in range(dim):
				C[i,k] = F

		for i in range(6):
			dt *= 0; dt[ (i+1) % dim] = 1; dt[i-1] = 1
			if np.array_equal(degree, dt):
				C[i,k] = 1

			dt *= 0; dt[i-2] = 1; dt[i-1] = 1
			if np.array_equal(degree, dt):
				C[i,k] = -1

			dt *= 0; dt[i] = 1
			if np.array_equal(degree, dt):
				C[i,k] = -1

	x0 = np.array([1 ] + [F,]*(dim-1))

	return phis, C, x0
		
 
