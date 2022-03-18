import numpy as np
from lide import *
from lide.lide2 import encode, decode	
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

if __name__ == '__main__':
	phis, C = lorenz63()
	dim = phis[0].dim
	x0 = np.array([-8,7,27])
	dt = 1e-2
	M = 1000
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])
	np.random.seed(0)
	noise = 2e0
	Ys = [X + noise*np.random.randn(*X.shape) for X in Xs]
	points = 5
	N = len(phis)

	Sigmas = [scipy.sparse.eye(Y.shape[0]*Y.shape[1]) for Y in Ys]
	W = 0*scipy.sparse.eye(N*dim)

	print("C true\n", C)

	lide = LIDE(phis, Ys, Sigmas, W, dts, points, mu = 1e-3, verbose = True, tol_dx = 1e-2)
	
	fig, ax =plt.subplots()
	ax.plot(X.T, '-')
	ax.set_prop_cycle(None)
	ax.plot(lide.Ys[0].T, '.')
	ax.set_prop_cycle(None)
	Xs0 = [savgol_filter(X, 15, 2) for X in Xs]
	ax.plot(Xs0[0].T, ':')

	#plt.show()
	#assert False
	lide.solve_threshold(maxiter = 1000, threshold = 1e-1, Xs0 = Xs0)
	#mask = np.abs(lide.C) > 1e-2
	#lide.solve_threshold(maxiter = 5, threshold = 1e-1, Xs0 = Xs0)
	print(lide.C)
	ax.set_prop_cycle(None)
	ax.plot(lide.Xs[0].T, '+')
	plt.show()
