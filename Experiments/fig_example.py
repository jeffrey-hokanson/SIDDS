
import numpy as np
from lide import *
import socket
from pgf import PGF
import scipy.sparse
import matplotlib.pyplot as plt

from joblib import Memory

if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_noise', verbose=0)
else:
	memory = Memory('.cache', verbose=0)

cache_evolve_system = memory.cache(evolve_system)

if __name__ == '__main__':
	dt = 1e-2
	M = 10000	
	phis, C, x0 = lorenz63()
	dim = phis[0].dim
	t = dt * np.arange(M)
	X = cache_evolve_system(phis, C, x0, t)

	Y = np.copy(X)
	np.random.seed(0)
	Y += 2*np.random.randn(*X.shape)

	oversample = 1	
	for phi, c in zip(phis, C.T):
		print(phi.degree, c)


	lide = LIDE(phis, [Y], [dt], verbose = True, oversample = oversample)
	lide.solve_irls(alpha = 5, q=0)
	for phi, c, c_true in zip(phis, lide.C.T, C.T):
		print(phi.degree, f'{c[0]:+5e}\t {c[1]:+5e}\t {c[2]:+5e} \t\t {c_true[0]:5f}\t {c_true[1]:5f}\t {c_true[2]:5f}')

	lide.solve_polish()
	for phi, c, c_true in zip(phis, lide.C.T, C.T):
		print(phi.degree, f'{c[0]:+5e}\t {c[1]:+5e}\t {c[2]:+5e} \t\t {c_true[0]:5f}\t {c_true[1]:5f}\t {c_true[2]:5f}')



	fig, axes = plt.subplots(3,2)
	for ax, x, y in zip(axes[:,0], X, lide.Xs[0]):
		ax.plot(t, x,)
		ax.plot(t, y,'.')


	pgf = PGF()
	pgf.add('t', t)
	pgf.add('x1', X[0])	
	pgf.add('x2', X[1])	
	pgf.add('x3', X[2])	
	pgf.add('y1', Y[0])	
	pgf.add('y2', Y[1])	
	pgf.add('y3', Y[2])	
	pgf.add('z1', lide.Xs[0][0])	
	pgf.add('z2', lide.Xs[0][1])	
	pgf.add('z3', lide.Xs[0][2])	
	pgf.write(f'data/fig_example_fit_over{oversample:d}.dat')

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
	pgf.write(f'data/fig_example_model_over{oversample:d}.dat')

	
	plt.show()
