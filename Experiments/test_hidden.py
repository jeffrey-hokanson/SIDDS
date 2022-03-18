import numpy as np
from lide import *
from scipy.signal import savgol_filter

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	phis, C = lorenz63()
	dim = phis[0].dim
	x0 = np.array([-8,7,27])
	dt = 1e-2
	M = 1000
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	dts = np.array([dt])
	np.random.seed(0)
	noise = 1e-4
	Y = np.copy(X)
	Y[0,:] = np.random.randn(M)
	Ys = [Y]
	
	points = 9

	data_mask = np.ones(Y.shape, dtype = bool)
	data_mask[0,:] = False

	s = data_mask + 1e-8* (~data_mask)	

	#S1 = 1e-6/dt*scipy.sparse.diags([-1,2, -1], offsets = [-1,0, 1], shape = (M, M), dtype = float)
	S1 = 0*scipy.sparse.eye(M)
	S2 = scipy.sparse.eye(M)
	S3 = scipy.sparse.eye(M)
	Sigma = scipy.sparse.block_diag([S1,S2,S3])
	Sigmas = [Sigma]
	print("C true\n", C)
	
	mask = (C != 0)
	lide = LIDE(phis, Ys, dts, points, mu = 1, verbose = True, tol_dx = 1e-5, mask = mask)
	C0 = 0*np.random.randn(*C.shape)
	Xsmooth = Ys
	for it in range(1):
		lide.solve( maxiter = 500, Xs0 = Xsmooth, C0 = C0)
		Xt = np.copy(lide.Xs[0])
		print(Xt - X)
		#print(Xt.shape)
		#Xt[0] = savgol_filter(Xt[0], 11, 1)
		#Xt[1:2,:] = X[1:2,:]
		Xt = savgol_filter(Xt, 31, 2)
		Xsmooth = [Xt]
		C0 = np.copy(lide.C)

		fig, ax =plt.subplots()
		ax.plot(X.T, '-')
		ax.set_prop_cycle(None)
		ax.plot(lide.Xs[0].T, '.')
		plt.show(block = False)
		plt.pause(1e-3)
		print(lide.C)
	plt.show()
	
	# initial solve
	if False:
		phis2 = []
		max_degree = 2
		for k, degree in enumerate(product(range(max_degree+1), repeat = 2)):
			phis2.append(Monomial(degree))
		N = len(phis)
		W = 0*scipy.sparse.eye(len(phis2)*2)
		Sigma = scipy.sparse.eye(M*2)
		lide = LIDE(phis2, [Y[1:,:]], [Sigma], W, dts, points, mu = 1, verbose = True, tol_dx =1e-5)
		lide.solve(maxiter = 10)

		fig, ax =plt.subplots()

		ax.plot(X.T, ':')
		ax.plot(lide.Xs[0].T)
		plt.show()
