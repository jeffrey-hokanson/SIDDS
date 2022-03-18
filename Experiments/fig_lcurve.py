import numpy as np
import scipy.linalg
import scipy.sparse
from lide import *
from pgf import PGF
from joblib import Memory, Parallel, delayed, parallel_backend
import socket

if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_lcurve', verbose=0)
else:
	memory = Memory('.cache', verbose=0)


def lide_irls(alpha, phis, Ys, dts, q = 0, points = 9):
	lide = LIDE(phis, Ys, dts, points = points, verbose = True)
	Xs0 = [denoise_corner(Y, dt) for Y, dt in zip(Ys, dts)] 
	lide.solve_irls(Xs0 = Xs0, alpha = alpha, q = q, epsilon_min = 1e-10)
	obj = lide.objective()
	reg = lide.regularization()
	C = lide.C
	nnz = np.sum(np.abs(C) > 1e-3)
	return obj, reg, nnz


def lide_irls_corner(phis, Ys, dts, q = 0):	
	phis, C, x0 = fun()
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])
	np.random.seed(0)
	Ys = [X + noise*np.random.randn(*X.shape) for X in Xs]
	cache_lide = memory.cache(lide_L_corner) 	
	
	lide_opt = cache_lide(phis, Ys, dts, q = q, alpha_min = 1e-5, alpha_max = 1e2, verbose = True)
	pgf = PGF()
	pgf.add('obj', [lide_opt.objective()])
	pgf.add('alpha', [lide_opt.alpha])
	pgf.add('reg', [lide_opt.regularization()])
	pgf.write(f'{prefix:s}_corner.dat')

def fig_lcurve(q = 0, dt = 1e-2, M = 1000, noise = 3e-1, alphas = np.logspace(-3, 2), epsilon0 = 1e0, fun = duffing, prefix = None):
	phis, C, x0 = fun()
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])

	np.random.seed(0)
	Ys = [X + noise*np.random.randn(*X.shape) for X in Xs]

	cache_lide = memory.cache(lide_irls) 	

	obj = np.nan * np.zeros(alphas.shape)
	reg = np.nan * np.zeros(alphas.shape)
	nnz = np.nan * np.zeros(alphas.shape)

	for i, alpha in enumerate(alphas):
		obj[i], reg[i], nnz[i] = cache_lide(alpha, phis, Ys, dts, q = q)
		pgf = PGF()
		pgf.add('alpha', alphas)
		pgf.add('obj', obj)
		pgf.add('reg', reg)
		pgf.add('nnz', nnz)
		pgf.write(prefix + '.dat')


if __name__ == '__main__':
	q = 0
	fun = duffing
	prefix = f'data/fig_lcurve_duffing_q{q:d}'
	noise = 1

	# Generate log-spaced samples
	alphas = [0]
	for k in range(-6,4):
		alphas += [10**k, 2*10**k, 5*10**k] 
	alphas += [.12,.15,.22,.25, .52, .55, .6, .61, .62, .63, .64, .65, .66, .67, .68, .69, .7, .8, .9]
	alphas = np.sort(np.array(alphas))

	if q == 0:
		fig_lcurve(q= q, alphas = alphas, fun = fun, prefix = prefix, noise = noise)

	elif q == 1:
		alphas = [0, 1e-7,3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,] + list(np.logspace(-2, 2, 30)) + [2e2,5e2,1e3]
		alphas = np.array(alphas)
		fig_lcurve(q= q, alphas = alphas, fun = fun, prefix = prefix, noise = noise)
