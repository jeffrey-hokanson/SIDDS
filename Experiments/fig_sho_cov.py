import numpy as np
from lide import *
import socket
from pgf import PGF
import scipy.sparse
import matplotlib.pyplot as plt

from joblib import Memory, delayed, Parallel, parallel_backend 

if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_noise', verbose=0)
else:
	memory = Memory('.cache', verbose=0)

cache_evolve_system = memory.cache(evolve_system)

@memory.cache
def fit_lide(seed, mask = None, noise = 1):
	dt = 1e-2
	M = 2000	
	phis, C, x0 = simple_harmonic_oscillator(include_constant = False)
	dim = phis[0].dim
	t = dt * np.arange(M)
	X = cache_evolve_system(phis, C, x0, t)
	
	Y = np.copy(X)
	np.random.seed(seed)
	Y += noise*np.random.randn(*X.shape)
	
	lide = LIDE(phis, [Y], [dt], verbose = False, points = 3, mask = mask)
	lide.solve(smooth = True)

	Cest = lide.C
	return vec(Cest)[vec(mask)]

@memory.cache
def fit_lsopinf(seed, mask = None, noise = 1):
	dt = 1e-2
	M = 2000	
	phis, C, x0 = simple_harmonic_oscillator(include_constant = False)
	dim = phis[0].dim
	t = dt * np.arange(M)
	X = cache_evolve_system(phis, C, x0, t)

	Y = np.copy(X)
	np.random.seed(seed)
	Y += noise*np.random.randn(*X.shape)

	Cest = lsopinf(phis, [Y], [dt], points = 3, mask = mask)
	return vec(Cest)[vec(mask)]



if __name__ == '__main__':
	N = 100_000
	phis, C, x0 = simple_harmonic_oscillator(include_constant = False)
	mask = np.abs(C)> 1e-4
	#mask = np.ones(C.shape, dtype = bool)
	#mask = None
	dim = phis[0].dim
	dt = 1e-2	
	t = dt * np.arange(2000)
	X = cache_evolve_system(phis, C, x0, t)

	c_true = vec(C)
	
	if False:
		CRB = cramer_rao(C, phis, [X], [dt], mask = mask )	
		P_C = scipy.sparse.eye(CRB.shape[0], np.sum(mask)).A
		cov_crb = P_C.T @ (CRB @ P_C)
		with open('data/fig_sho_cov_CRB.csv','w') as f:
			for cov_row in cov_crb:
				f.write(','.join([f'{c:5e}' for c in cov_row]) + '\n' )
		print("=====CRB")
		print(cov_crb)
		#import sys
		#sys.exit()

	if True:
		delayed_fit = delayed(fit_lsopinf)
		fname = 'data/fig_sho_cov_lsopinf.csv'
	else:
		delayed_fit = delayed(fit_lide)
		fname = 'data/fig_sho_cov_lide.csv'

	noise = 1e-2
	jobs = [delayed_fit(seed, mask = mask, noise = noise) for seed in range(N)]
	with parallel_backend('loky', inner_max_num_threads = 1, n_jobs =  -1): 
		c_vecs = Parallel(n_jobs = -1, verbose = 100, backend = 'loky')(jobs)


	dc = np.sum(mask)
	E_c = np.zeros(dc)
	E_c = np.sum(np.array(c_vecs), axis = 0)/len(c_vecs)
	print("expected value", E_c)
	cov = np.zeros((dc, dc))
	for c in c_vecs:
		cov += np.outer(c - E_c, c - E_c)
	cov /= N
	print("covariance")
	print(cov/noise**2)

	with open(fname,'w') as f:
		for cov_row in cov/noise**2:
			f.write(','.join([f'{c:5e}' for c in cov_row]) + '\n' )



	
		
	
