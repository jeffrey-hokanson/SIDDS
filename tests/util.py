import numpy as np
import lide

def check_grad(func, grad, x0, epsilon = 1e-7):
	n = len(x0)
	J = grad(x0)
	Jest = np.zeros_like(J)

	for i in range(n):
		ei = np.zeros(n)
		ei[i] = 1.
		fp = func(x0 + ei*epsilon)
		fn = func(x0 - ei*epsilon)
		Jest[...,i] = (fp - fn)/(2*epsilon)

	for i in range(n):
		print("J   ", J[...,i])
		print("Jest", Jest[...,i])
		print("err row", np.max(np.abs(J[...,i]-Jest[...,i])))
	return np.linalg.norm( (J - Jest).flatten())

def generate_data(fun = lide.duffing, noise = 1e-5, T = 500, M = 20, seed = 0, dt = 1e-4):
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

	return Ys, phis, dts, C
