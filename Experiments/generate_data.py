
import numpy as np
import lide

def generate_data(fun = lide.duffing, noise = 1e-2, T = 1, M = 1000, seed = 0, dt = 1e-2):
	r""" Generate test data

	"""
	# Generate test data
	phis, C, x0 = fun()
	dim = phis[0].dim
	np.random.seed(seed)
	Xs = []
	dts = []
	for i in range(T):
		if i > 0:
			x0 = np.random.uniform(0,1, size = (dim,) )
		t = dt*np.arange(M)
		X = lide.evolve_system(phis, C, x0, t)
		Xs.append(X)
		dts.append(dt)

	Ys = [X + noise*np.random.randn(*X.shape) for X in Xs]

	return Xs, Ys, phis, C, dts

def generate_lorenz_data(noise = 1e-2, T = 1, M = 1000, seed = 0, dt = 1e-2):
	assert T == 1
	np.random.seed(seed)
	phis, C = lide.lorenz63()
	x0 = np.array([-8, 7,27])
	t = dt*np.arange(M)
	X = lide.evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = [dt]
	Ys = [X + noise*np.random.randn(*X.shape)]

	return Xs, Ys, phis, C, dts
