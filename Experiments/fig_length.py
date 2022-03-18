import numpy as np
from lide.examples import lorenz63
from lide import evolve_system
from pgf import PGF
from functools import partial

from fig_noise import *
from fig_correlated import compute_cramer_rao

import argparse

from joblib import Memory, Parallel, delayed, parallel_backend
if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_noise', verbose=0)
else:
	memory = Memory('.cache', verbose=0)


def length_study_lorenz63(fname, fit, lengths, Ntrials = 10, 
	points = 9, dt = 1e-2, noise = 1e-2, n_jobs = -1, only = None,  **kwargs):
	
	phis, C, x0 = lorenz63()
	dim = phis[0].dim
	max_M = max(lengths)
	t = dt * np.arange(max_M)
	cache_evolve = memory.cache(evolve_system)
	X = cache_evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = [dt]
	

	jobs = []
	delayed_fit = delayed(memory.cache(fit))

	for i, length in enumerate(lengths):
		for j, seed in enumerate(range(Ntrials)):
			np.random.seed(seed)
			Ys = [X[:,:length] + noise * np.random.randn(dim, length) for X in Xs]
			jobs.append(delayed_fit(Ys, phis, dts, points, **kwargs))		
	
	if only is None:
		print("running parallel job")
		with parallel_backend('loky', inner_max_num_threads = 1, n_jobs = n_jobs): 
			job_results = Parallel(n_jobs = n_jobs, verbose = 100, backend = 'loky')(jobs)
	else:
		Parallel(n_jobs = 1, verbose = 100, backend = 'loky')(jobs[only])
		return	

	err = np.nan*np.zeros((len(lengths), Ntrials,))
	idx = 0
	for i, length in enumerate(lengths):
		for j, seed in enumerate(range(Ntrials)):
			Cest = job_results[idx]
			idx += 1
			try:
				err[i,j] = np.linalg.norm(C - Cest, 'fro')
			except ValueError:
				Cest = Cest.T
				err[i,j] = np.linalg.norm(C - Cest, 'fro')

	pgf = PGF()
	pgf.add('length', lengths)
	p0, p25, p50, p75, p100 = np.nanpercentile(err, [0, 25, 50, 75, 100], axis = 1)
	pgf.add('F0', p0)
	pgf.add('F25', p25)
	pgf.add('F50', p50)
	pgf.add('F75', p75)
	pgf.add('F100', p100)
	

	print("Computing CR Bound")

	if False:
		delayed_cr = delayed(memory.cache(compute_cramer_rao))
		jobs = []
		for M in lengths:	
			jobs.append(delayed_cr(C, phis, [X[:,:M]], dts,))
		
		with parallel_backend('loky', inner_max_num_threads = 1): 
			job_results = Parallel(n_jobs = -1, verbose = 100, backend = 'loky')(jobs)
		print("running parallel job complete")
		
		crb  = [noise*cr for cr in job_results]
	else:
		crb = np.nan*np.zeros(len(lengths))
		for i, m in enumerate(lengths):
			try:
				crb[i] = noise * float(compute_cramer_rao(C, phis, [X[:,:m]], dts, progress = True, atol = 1e-8, rtol = 1e-8))
			except Exception as e:
				print(e)
				break
			print('m', m, 'crb', crb[i])

	pgf.add('crb', crb)
	pgf.write(fname)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('alg')
	parser.add_argument('--only', type = int, default = None)
	args = parser.parse_args()

	if args.alg == 'lide':
		fname = 'data/fig_length_lorenz63_lide.dat'
		fit = fit_lide_smooth
		kwargs = {'solve_qp_kwargs': {'maxiter': 2000,}}  
	elif args.alg == 'lsopinf':
		fit = fit_lsopinf 
		fname = 'data/fig_length_lorenz63_lsopinf.dat'
		kwargs = {}
	else:
		raise NotImplementedError


	lengths = np.logspace(2, 5, 3*24+1, dtype = int)
	if args.only is None:
		length_study_lorenz63(fname, fit, lengths, Ntrials = 100, n_jobs = -1, **kwargs)
	else:
		length_study_lorenz63(fname, fit, lengths, Ntrials = 100, n_jobs = 1, only = args.only, **kwargs)
	
