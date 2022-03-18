# An experiment with correlated noise
import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
from fig_noise import *
from joblib import Memory, Parallel, delayed, parallel_backend
from copy import deepcopy as copy

if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_correlated', verbose=0)
else:
	memory = Memory('.cache', verbose=0)


def estimate_median(cov_C, N = 100_000):
	np.random.seed(0)
	L = np.linalg.cholesky(cov_C)
	Z = np.random.randn(cov_C.shape[1], N)
	LZ = L @ Z
	l2_norm = np.sqrt(np.sum(LZ**2, axis = 0))
	return np.percentile( l2_norm, [50])

def compute_cramer_rao(C, phis, Xs, dts, Ms = None, **kwargs):
	cov = cramer_rao(C, phis, Xs, dts, Ms = Ms, **kwargs)
	
	d, m = C.shape
	P_C = scipy.sparse.eye(cov.shape[0], C.shape[0] * C.shape[1]).A
	cov_C =  P_C.T @ (cov @ P_C) 
	return estimate_median(cov_C)


def correlation_study(C, phis, Xs, dts, points, correlations = [0], noise = 1e-3, Ntrials = 10, fit = fit_lide, fname = None, **kwargs):
	assert fname is not None, "must provide a file name"

	C_mask = np.abs(C) > 0
	tol = 0.1*np.min(np.abs(C[C_mask]))

	jobs = []
	delayed_fit = delayed(memory.cache(fit))
	n = Xs[0].shape[0]
	

	all_invSigma = []
	for i, cor in enumerate(correlations):
		for j, seed in enumerate(range(Ntrials)):
			np.random.seed(seed)
			if n == 2:
				Sigma = np.array([[1, cor],[cor, 1]])
			elif n == 3:
				Sigma = np.array([[1, cor, 0],[cor, 1, 0], [0,0,1]])
			else:	
				raise NotImplementedError
			transform = np.linalg.cholesky(Sigma)
			inv_transform = scipy.linalg.solve_triangular(transform, np.eye(n), lower = True)
			invSigma = scipy.linalg.inv(Sigma)
			Ns = [noise * transform @ np.random.randn(*X.shape) for X in Xs]
			Ys = [X + N for X, N in zip(Xs, Ns)]
			Sigmas = []
			for X in Xs:
				Sigma1 = scipy.sparse.kron(Sigma, scipy.sparse.eye(X.shape[1]))
				invSigma2 = scipy.sparse.kron(scipy.sparse.eye(X.shape[1]), invSigma)
				Sigmas.append(invSigma2)
			norm = np.sum([vec(N).T @ Sigma @ vec(N) for Sigma, N in zip(Sigmas, Ns)])
			#print(f'seed {seed}, mismatch {norm:5e}')
			kwargs['Sigmas'] = Sigmas
			if seed == 0:
				all_invSigma.append(copy(Sigmas))
			jobs.append(delayed_fit(Ys, phis, dts, points, **kwargs))

	print("running parallel job")
	with parallel_backend('loky', inner_max_num_threads = 1): 
		job_results = Parallel(n_jobs = -1, verbose = 100, backend = 'loky')(jobs)
	print("running parallel job complete")
	
	jaccard_dist = np.nan*np.zeros((len(correlations), Ntrials))
	err = np.nan*np.zeros((len(correlations), Ntrials))
	idx = 0
	for i, cor in enumerate(correlations):
		for j, seed in enumerate(range(Ntrials)):
			Cest = job_results[idx]
			idx += 1
			#print("Cest\n", Cest)
			#print("C\n", C)
			err[i,j] = np.linalg.norm(C - Cest, 'fro')
			Cest_mask = np.abs(Cest)>tol
			#for k,l in np.ndindex(*C.shape):
			#	print(C[k,l], Cest[k,l], C_mask[k,l], Cest_mask[k,l])
			jaccard_dist[i,j] = jaccard(C_mask.flatten(), np.abs(Cest.flatten()) > tol)
			print("cor", cor, "seed", seed, "error", err[i,j], "jaccard", jaccard_dist[i,j])			
			
	pgf = PGF()
	pgf.add('correlation', correlations)
	p0, p25, p50, p75, p100 = np.nanpercentile(err, [0, 25, 50, 75, 100], axis = 1)
	pgf.add('F0', p0)
	pgf.add('F25', p25)
	pgf.add('F50', p50)
	pgf.add('F75', p75)
	pgf.add('F100', p100)
	p0, p25, p50, p75, p100 = np.nanpercentile(jaccard_dist, [0, 25, 50, 75, 100], axis = 1)
	pgf.add('J0', p0)
	pgf.add('J25', p25)
	pgf.add('J50', p50)
	pgf.add('J75', p75)
	pgf.add('J100', p100)

	
	print("Cramer-Rao")
	jobs = []
	delayed_cr = delayed(memory.cache(compute_cramer_rao))
	#print( (all_invSigma[0][0] - all_invSigma[-1][0]).A)
	for Ms in all_invSigma:
		#print( (Ms[0] - all_invSigma[0][0]).A)
		jobs.append(delayed_cr(C, phis,  Xs, dts, Ms = Ms))

	with parallel_backend('loky', inner_max_num_threads = 1): 
		job_results = Parallel(n_jobs = -1, verbose = 100, backend = 'loky')(jobs)
	print("running parallel job complete")

	crb  = [noise*cr for cr in job_results]
	pgf.add('crb', crb)
	pgf.write(fname)


def ex_duffing(M = int(10/1e-2), dt = 1e-2, points = 9, Ntrials = 100, noise = 1e-2, fit = None, fname = None,  **kwargs):	
	phis, C, x0 = duffing()
	dim = phis[0].dim
	
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])

	correlations = np.array(list(0.02*np.arange(0,50)) + [0.99, 0.991, 0.992, 0.995, 0.999])
	correlation_study(C, phis, Xs, dts, points, noise = noise,
		correlations = correlations, Ntrials = Ntrials, fit = fit, fname = fname, **kwargs)



def ex_vanderpol(M = int(10/1e-2), dt = 1e-2, points = 9, Ntrials = 100, noise = 1e-2, fit = None, fname = None,  **kwargs):	
	phis, C, x0 = vanderpol()
	dim = phis[0].dim
	
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])

	correlations = np.array(list(0.02*np.arange(0,50)) + [0.99, 0.991, 0.992, 0.995, 0.999])
	correlation_study(C, phis, Xs, dts, points, noise = noise,
		correlations = correlations, Ntrials = Ntrials, fit = fit, fname = fname, **kwargs)


def ex_lorenz63(M = int(20/1e-2), dt = 1e-2, points = 9, Ntrials = 100, noise = 1e-2, fit = None, fname = None,  **kwargs):	
	phis, C, x0 = lorenz63()
	dim = phis[0].dim
	
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])

	correlations = np.array(list(0.02*np.arange(0,50)) + [0.99, 0.991, 0.992, 0.995, 0.999])
	correlation_study(C, phis, Xs, dts, points, noise = noise,
		correlations = correlations, Ntrials = Ntrials, fit = fit, fname = fname, **kwargs)




if __name__ == '__main__':
	
	Ntrials = 100
	
	#ex_duffing( fit = fit_lide, fname = 'data/fig_correlated_duffing_lide.dat', Ntrials = 100, noise = 1e-2)	
	ex_duffing( fit = fit_lsopinf, fname = 'data/fig_correlated_duffing_lsopinf.dat', Ntrials = 100, noise = 1e-2)	
	

	#ex_lorenz63( fit = fit_lide, fname = 'data/fig_correlated_lorenz63_lide.dat', Ntrials = Ntrials, noise = 1e-2)	
	ex_lorenz63( fit = fit_lsopinf, fname = 'data/fig_correlated_lorenz63_lsopinf.dat', Ntrials = Ntrials, noise = 1e-2)	

	#ex_vanderpol( fit = fit_lide, fname = 'data/fig_correlated_vanderpol_lide.dat', Ntrials = Ntrials, noise = 1e-2)	
	ex_vanderpol( fit = fit_lsopinf, fname = 'data/fig_correlated_vanderpol_lsopinf.dat', Ntrials = Ntrials, noise = 1e-2)	
	
