import numpy as np
from lide import *
import socket
from pgf import PGF
from joblib import Memory
import scipy.sparse
import matplotlib.pyplot as plt

if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_noise', verbose=0)
else:
	memory = Memory('.cache', verbose=0)

cache_evolve_system = memory.cache(evolve_system)

if __name__ == '__main__':
	dt = 1e-2
	M = 2000	
	phis, C, x0 = lorenz63()
	dim = phis[0].dim
	t = dt * np.arange(M)
	X = cache_evolve_system(phis, C, x0, t)

	Y = np.copy(X)
	np.random.seed(0)
	#Y[0] = np.max(np.abs(X))*np.random.randn(M)
	Y[0] = 0
	#Y[0] += 1 
	d = np.ones(dim*M)
	d[::3] = 1e-3
	Sigma = scipy.sparse.diags(d)
	
	
	for phi, c in zip(phis, C.T):
		print(phi.degree, c)

#	obj = LIDEObjective(phis, [Y], [Sigma], [dt], points = 3)
#	f = obj.fun(obj.encode(C, [X]))	
#	print(f)
#	assert False


	lide = LIDE(phis, [Y], [dt], Sigmas = [Sigma], verbose = True)
	lide.solve_irls(Xs0 = [Y],  alpha = 1, maxiter = 80,  q = 1)

	print("True")
	print(C)
	print("Recovered")
	print(lide.C)
	fig, axes = plt.subplots(3,2)
	for ax, x, y in zip(axes[:,0], X, lide.Xs[0]):
		ax.plot(t, x,)
		ax.plot(t, y,'.')


	pgf = PGF()
	pgf.add('t', t)
	pgf.add('x1', X[0])	
	pgf.add('x2', X[1])	
	pgf.add('x3', X[2])	
	pgf.add('y1', lide.Xs[0][0])	
	pgf.add('y2', lide.Xs[0][1])	
	pgf.add('y3', lide.Xs[0][2])	
	pgf.write('data/fig_hidden_fit.dat')

	err = np.linalg.norm(lide.Xs[0][1:] - X[1:], 'fro')**2
	print(f'mismatch on observed data {err:5e}')
	mis = vec(lide.Xs[0] - X)
	obj = mis.T @ Sigma @ mis
	print(f'objective                 {obj:5e}')

	t = dt * np.arange(2*M)
	Xp = evolve_system(phis, C, x0, t)
	x0 = lide.Xs[0][:,0]
	Yp = evolve_system(phis, lide.C, x0, t)
	for ax, x, y in zip(axes[:,1], Xp, Yp):
		ax.plot(t, x,)
		ax.plot(t, y,'.')
	
	pgf = PGF()
	pgf.add('t', t)
	pgf.add('x1', Xp[0])	
	pgf.add('x2', Xp[1])	
	pgf.add('x3', Xp[2])	
	pgf.add('y1', Yp[0])	
	pgf.add('y2', Yp[1])	
	pgf.add('y3', Yp[2])	
	pgf.write('data/fig_hidden_predict.dat')
	
	pgf = PGF()
	pgf.add('d1', [phi.degree[0] for phi in phis])
	pgf.add('d2', [phi.degree[1] for phi in phis])
	pgf.add('d3', [phi.degree[2] for phi in phis])
	pgf.add('c1', C[0])
	pgf.add('c2', C[1])
	pgf.add('c3', C[2])
	pgf.add('cest1', lide.C[0])
	pgf.add('cest2', lide.C[1])
	pgf.add('cest3', lide.C[2])
	pgf.write('data/fig_hidden_model.dat')

	for phi, c, c_true in zip(phis, lide.C.T, C.T):
		print(phi.degree, f'{c[0]:+5e}\t {c[1]:+5e}\t {c[2]:+5e}')
	
	plt.show()
