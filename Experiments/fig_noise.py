import numpy as np
from lide import *
from lide.modified_sindy import *
from lide.sindy import *
from pgf import PGF
from scipy.spatial.distance import jaccard
from joblib import Memory, Parallel, delayed, parallel_backend
import socket
import _pickle as cPickle
import argparse
from copy import deepcopy as copy

LOKY_PICKLER = cPickle

if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_noise', verbose=0)
else:
	memory = Memory('.cache', verbose=0)
	
cache_evolve_system = memory.cache(evolve_system)

def fit_sindy(Ys, phis, dts, points, **kwargs):
	kwargs = copy(kwargs)
	for key in kwargs:
		if key not in ['threshold', 'alpha']:
			del kwargs[key]

	return sindy(phis, Ys, dts, **kwargs)  

def fit_lsopinf(Ys, phis, dts, points, **kwargs):
	return lsopinf(phis, Ys, dts, points)

def fit_mod_sindy(Ys, phis, dts, points, **kwargs):
	C, Xs = modified_SINDy(phis, Ys, dts, **kwargs)
	return C

def fit_lide(Ys, phis, dts, points, mu = 1e2, maxiter = 100, kkt_reg = 1e-4, **kwargs):
	lide = LIDE(phis, Ys, dts, points = points, mu = mu, kkt_reg = kkt_reg, verbose = True, tol_opt = 1e-8, tol_dx = 1e-8, **kwargs)
	lide.solve(maxiter = maxiter)
	return lide.C

def fit_lide_smooth(Ys, phis, dts, points, mu = 1e2, maxiter = 100, kkt_reg = 1e-4, smooth = 1e-7, **kwargs):
	# Use initial smoothing with a fixed value
	lide = LIDE(phis, Ys, dts, points = points, mu = mu, kkt_reg = kkt_reg, verbose = True, tol_opt = 1e-8, tol_dx = 1e-8, **kwargs)
	lide.solve(maxiter = maxiter, smooth = smooth)
	return lide.C

def fit_lide_irls0(Ys, phis, dts, points, **kwargs):
	lide = lide_L_corner(phis, Ys, dts, q = 0, points = points, verbose = False, **kwargs)
	return lide.C

def fit_lide_irls0_fixed(Ys, phis, dts, points, mu = 1e2, kkt_reg = 1e-4, alpha = 1e-2, epsilon0 = 1e0, epsilon_min = 1e-8, **kwargs):
	lide = LIDE(phis, Ys, dts, points = points, mu = mu, kkt_reg = kkt_reg, verbose = True)
	lide.solve_irls(alpha = alpha, q = 0, epsilon = epsilon0, epsilon_min = epsilon_min, maxiter = 40)
	return lide.C

def fit_lide_irls0_fixed_smooth(Ys, phis, dts, points, mu = 1e2, kkt_reg = 1e-4, alpha = 1e-2, epsilon0 = 1e0, epsilon_min = 1e-8, smooth = 1e-7, **kwargs):
	lide = LIDE(phis, Ys, dts, points = points, mu = mu, kkt_reg = kkt_reg, verbose = True)
	lide.solve_irls(alpha = alpha, q = 0, epsilon = epsilon0, epsilon_min = epsilon_min, maxiter = 40, smooth = smooth)
	return lide.C

def fit_lide_irls0_fixed_smooth_polish(Ys, phis, dts, points, mu = 1e2, kkt_reg = 1e-4, alpha = 1e-2, epsilon0 = 1e0, epsilon_min = 1e-8, smooth = 1e-7, **kwargs):
	lide = LIDE(phis, Ys, dts, points = points, mu = mu, kkt_reg = kkt_reg, verbose = True)
	lide.solve_irls(alpha = alpha, q = 0, epsilon = epsilon0, epsilon_min = epsilon_min, maxiter = 40)
	nnz = int(np.round(lide.regularization()))
	threshold = np.sort(np.abs(lide.C.flatten()))[-nnz]
	mask = np.abs(lide.C) >= threshold
	lide.tol_opt = 1e-8	
	lide.solve(C0 = lide.C * mask, mask = mask, maxiter = 100)
	return lide.C

def fit_lide_irls1(Ys, phis, dts, points, **kwargs):
	lide = lide_L_corner(phis, Ys, dts, q = 1, points = points **kwargs) 
	return lide.C

def noise_study(C, phis, Xs, dts, points, noises = [1e-3], Ntrials = 10, fit = fit_lide, fname = None, n_jobs = -1, only = None, **kwargs):
	assert fname is not None, "must provide a file name"
	jaccard_dist = np.nan*np.zeros((len(noises), Ntrials))
	err = np.nan*np.zeros((len(noises), Ntrials))

	C_mask = np.abs(C) > 0
	tol = 0.1*np.min(np.abs(C[C_mask]))

	jobs = []
	delayed_fit = delayed(memory.cache(fit))
	for i, noise in enumerate(noises):
		for j, seed in enumerate(range(Ntrials)):
			np.random.seed(seed)
			Ys = [X + noise*np.random.randn(*X.shape) for X in Xs]
			jobs.append(delayed_fit(Ys, phis, dts, points, **kwargs))

	if only:
		print("running jobs", only, "of", len(jobs))
		jobs = [jobs[i] for i in only]

	print("running parallel job")
	with parallel_backend('loky', inner_max_num_threads = 1, n_jobs = n_jobs): 
		job_results = Parallel(n_jobs = n_jobs, verbose = 100, backend = 'loky')(jobs)
	print("running parallel job complete")
	if only:	
		return 
	
	idx = 0
	for i, noise in enumerate(noises):
		for j, seed in enumerate(range(Ntrials)):
			Cest = job_results[idx]
			idx += 1
			#print("Cest\n", Cest)
			#print("C\n", C)
			try:
				err[i,j] = np.linalg.norm(C - Cest, 'fro')
			except ValueError:
				Cest = Cest.T
				err[i,j] = np.linalg.norm(C - Cest, 'fro')
				
			Cest_mask = np.abs(Cest)>tol
			#for k,l in np.ndindex(*C.shape):
			#	print(C[k,l], Cest[k,l], C_mask[k,l], Cest_mask[k,l])
			jaccard_dist[i,j] = jaccard(C_mask.flatten(), np.abs(Cest.flatten()) > tol)
			print("noise", noise, "seed", seed, "error", err[i,j], "jaccard", jaccard_dist[i,j])			
			
	pgf = PGF()
	pgf.add('noise', noises)
	p0, p25, p50, p75, p100 = np.nanpercentile(err, [0, 25, 50, 75, 100], axis = 1)
	pgf.add('F0', p0)
	pgf.add('F25', p25)
	pgf.add('F50', p50)
	pgf.add('F75', p75)
	pgf.add('F100', p100)
	pgf.add('mean', np.mean(err, axis = 1))
	p0, p25, p50, p75, p100 = np.nanpercentile(jaccard_dist, [0, 25, 50, 75, 100], axis = 1)
	pgf.add('J0', p0)
	pgf.add('J25', p25)
	pgf.add('J50', p50)
	pgf.add('J75', p75)
	pgf.add('J100', p100)

	# exact recovery fraction
	success_trials = np.sum(np.isfinite(jaccard_dist), axis = 1)
	frac = np.sum(jaccard_dist <=1e-7, axis = 1) / success_trials
	pgf.add('exact_sparse', frac)
	pgf.add('successful_trials', success_trials)
	pgf.write(fname)
						

	
def ex_duffing(M = int(10/1e-2), dt = 1e-2, points = 9, Ntrials = 100, fit = None, fname = None, **kwargs):	
	phis, C, x0 = duffing()
	dim = phis[0].dim
	
	t = dt * np.arange(M)
	X = cache_evolve_system(phis, C, x0, t)

	pgf = PGF()
	pgf.add('x', X[0])
	pgf.add('y', X[1])
	pgf.write('data/fig_noise_duffing_state.dat')

	Xs = [X]
	dts = np.array([dt])

	noises = 10**( -3 + 0.1*np.arange(0, 31))
		
	noise_study(C, phis, Xs, dts, points, noises, Ntrials, fit, fname = fname, **kwargs)

def ex_vanderpol(M = int(10/1e-2), dt = 1e-2, points = 9, Ntrials = 100, fit = None, fname = None, **kwargs):	
	phis, C, x0 = vanderpol()
	dim = phis[0].dim
	
	t = dt * np.arange(M)
	X = cache_evolve_system(phis, C, x0, t)
	pgf = PGF()
	pgf.add('x', X[0])
	pgf.add('y', X[1])
	pgf.write('data/fig_noise_vanderpol_state.dat')
	

	Xs = [X]
	dts = np.array([dt])

	noises = 10**( -2 + 0.1*np.arange(0, 31))
		
	noise_study(C, phis, Xs, dts, points, noises, Ntrials, fit, fname = fname, **kwargs)

def ex_lorenz63(M = int(20/1e-2), dt = 1e-2, points = 9, Ntrials = 100, fit = None, fname = None, **kwargs):	
	phis, C, x0 = lorenz63()
	dim = phis[0].dim
	t = dt * np.arange(M)
	X = cache_evolve_system(phis, C, x0, t)
	pgf = PGF()
	pgf.add('x', X[0])
	pgf.add('y', X[1])
	pgf.add('z', X[2])
	pgf.write('data/fig_noise_lorenz63_state.dat')
	
	Xs = [X]
	dts = np.array([dt])

	noises = 10**( -2 + 0.1*np.arange(0, 31))
	noise_study(C, phis, Xs, dts, points, noises, Ntrials, fit, fname = fname, **kwargs)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('case')
	parser.add_argument('alg', default = 'all')
	parser.add_argument('--trials', type = int, default = 100)
	parser.add_argument('--jobs', type = int, default = -1)
	parser.add_argument('--alpha', type = float, default =1e-2)
	parser.add_argument('--only', type = int, default =None, nargs = '*')
	args = parser.parse_args()
	
	name = f'data/fig_noise_{args.case}_{args.alg}.dat'
	kwargs = {'only': args.only}

	if args.alg == 'lsopinf':
		fit = fit_lsopinf
		kwargs['points'] = 3
	elif args.alg == 'sindy':
		fit = fit_sindy
		name = f'data/fig_noise_{args.case}_sindy_alpha_{args.alpha}.dat'
		kwargs['alpha'] = args.alpha
	elif args.alg == 'lide':
		fit = fit_lide
	elif args.alg == 'lide_smooth':
		fit = fit_lide_smooth
	elif args.alg == 'lide_irls0':
		fit = fit_lide_irls0
	elif args.alg == 'lide_irls0_fixed':
		fit = fit_lide_irls0_fixed
		name = f'data/fig_noise_{args.case}_lide_irls0_alpha_{args.alpha}.dat'
		kwargs['alpha'] = args.alpha
	elif args.alg == 'lide_irls0_fixed_smooth':
		fit = fit_lide_irls0_fixed_smooth
		name = f'data/fig_noise_{args.case}_lide_irls0_smooth_alpha_{args.alpha}.dat'
		kwargs['alpha'] = args.alpha
	elif args.alg == 'lide_irls0_fixed_smooth_polish':
		fit = fit_lide_irls0_fixed_smooth_polish
		name = f'data/fig_noise_{args.case}_lide_irls0_smooth_polish_alpha_{args.alpha}.dat'
		kwargs['alpha'] = args.alpha
	elif args.alg == 'mod_sindy':
		fit = fit_mod_sindy
		kwargs['noise_init'] = True
		if args.case == 'duffing':
			kwargs['lam'] = 0.05
			kwargs['n_train'] = 5000
		elif args.case == 'lorenz63':
			kwargs['lam'] = 0.2
			kwargs['n_train'] = 15000
		elif args.case == 'vanderpol':
			kwargs['lam'] = 0.1 
			kwargs['n_train'] = 5000


	if args.case == 'lorenz63':
		ex_lorenz63(fit = fit, fname = name, n_jobs = args.jobs, Ntrials = args.trials, **kwargs)
	elif args.case == 'duffing':
		ex_duffing(fit = fit, fname = name, n_jobs = args.jobs, Ntrials = args.trials, **kwargs)
	elif args.case == 'vanderpol':
		ex_vanderpol(fit = fit, fname = name, n_jobs = args.jobs, Ntrials = args.trials, **kwargs)
			
