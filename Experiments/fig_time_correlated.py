# An experiment with correlated noise
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from fig_noise import *
from lide.matrices import make_vec
import matplotlib.pyplot as plt

from joblib import Memory, Parallel, delayed, parallel_backend

if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_time_correlated', verbose=0)
else:
	memory = Memory('.cache', verbose=0)


def fit_lide(Ys, phis, dts, points, Sigmas = None, mu = 1e2, maxiter = 100, kkt_reg = 1e-4, **kwargs):
	lide = LIDE(phis, Ys, dts, Sigmas = Sigmas, points = points, mu = mu, kkt_reg = kkt_reg, verbose = True, 
			solver ='dense', **kwargs)
	lide.solve(maxiter = maxiter)
	return lide.C

def fit_lsopinf(Ys, phis, dts, points, **kwargs):
	return lsopinf(phis, Ys, dts, points)


def correlation_study(C, phis, Xs, dts, points, correlations = [0], noise = 1e-3, Ntrials = 10, fit = fit_lide, fname = None, n_jobs = -1, **kwargs):
	assert fname is not None, "must provide a file name"
	jaccard_dist = np.nan*np.zeros((len(correlations), Ntrials))
	err = np.nan*np.zeros((len(correlations), Ntrials))

	C_mask = np.abs(C) > 0
	tol = 0.1*np.min(np.abs(C[C_mask]))
	assert len(dts) == 1, "Currently only configured for a single series"

	jobs = []
	delayed_fit = delayed(memory.cache(fit))
	for i, cor in enumerate(correlations):	
		n, M = Xs[0].shape
		t = dts[0]*np.arange(M)
		if cor > 0:
			r = np.exp(-1*(t/cor)**2)
		else:
			r = np.ones(1)
		#r = r[r > 1e-16]
		print("r", r[0:5])
		data = np.hstack([r[:0:-1], r])
		diags = np.hstack([np.arange(-len(r)+1, 0), np.arange(0, len(r))])
		Sigma = scipy.sparse.diags(data, diags, shape = (M, M) )
		#print(Sigma.A[0:5,0:5])
		U, s, VT = scipy.linalg.svd(Sigma.A, full_matrices = False)
		L = U @ np.diag(s**0.5) @ VT
		#print("singular values", s)
		#invSigma = scipy.sparse.linalg.inv(Sigma.tocsr())
		invSigma = U @ np.diag(1./s) @ VT
		#fig, ax = plt.subplots(2)
		#ax[0].spy(np.abs(Sigma) > 1e-16)
		#ax[1].spy(np.abs(invSigma) > 1e-10)
		plt.show()
		vec_invSigma = make_vec(invSigma, n)
		vec_Sigma = make_vec(Sigma, n)
		vec_L = make_vec(L, n)
		#vec_L = scipy.linalg.cholesky(vec_Sigma.A)
		#print(np.linalg.norm(vec_L1.A - vec_L, 'fro'))	

		for j, seed in enumerate(range(Ntrials)):
			np.random.seed(seed)
			N = unvec(noise * vec_L @ np.random.randn(M*n), n)
			Ys = [Xs[0] + N]
			kwargs['Sigmas'] = [vec_invSigma]
			norm = vec(N).T @ vec_invSigma @ vec(N)
			print(f'norm {norm:20.15e}')
			jobs.append(delayed_fit(Ys, phis, dts, points, **kwargs))



	print("running parallel job")
	with parallel_backend('loky', inner_max_num_threads = 1): 
		job_results = Parallel(n_jobs = n_jobs, verbose = 100, backend = 'loky')(jobs)
	print("running parallel job complete")
	
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
	pgf.write(fname)


def ex_duffing(M = int(10/1e-2), dt = 1e-2, points = 9, Ntrials = 100, noise = 1e-2, fit = None, fname = None,  **kwargs):	
	phis, C, x0 = duffing()
	dim = phis[0].dim
	
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])

	#correlations = 0.02*np.arange(1,50)
	#correlations = [0,0.9, 0.99, 0.999]
	correlations = 0.001*np.arange(0, 30)
	correlation_study(C, phis, Xs, dts, points, noise = noise,
		correlations = correlations, Ntrials = Ntrials, fit = fit, fname = fname, **kwargs)

if __name__ == '__main__':
	
	ex_duffing( fit = fit_lide, fname = 'data/fig_time_correlated_duffing_lide.dat', Ntrials = 100, noise = 1e-3, n_jobs = -1)	
	#ex_duffing( fit = fit_lsopinf, fname = 'data/fig_time_correlated_duffing_lsopinf.dat', Ntrials = 100, noise = 1e-3)	
