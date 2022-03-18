import numpy as np
from lide import *
	

if __name__ == '__main__':
	phis, C = lorenz63()
	dim = phis[0].dim
	x0 = np.array([-8,7,27])
	dt = 1e-2
	M = 2000
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])
	np.random.seed(0)
	noise = 1e-1
	Ys = [X + noise*np.random.randn(*X.shape) for X in Xs]
	points = 3
	N = len(phis)
	Sigmas = [scipy.sparse.eye(Y.shape[0]*Y.shape[1]) for Y in Ys]
	W = 0*scipy.sparse.eye(N*dim)

	print("C true\n", C)
	
	lide = LIDE(phis, Ys, Sigmas, W, dts, points, mu = 1, verbose = True, tol_dx = 1e-6)
	lide.solve(q = 0, rho = 4e-3, maxiter = 50, rank_est = 7)
	print(lide.C)
