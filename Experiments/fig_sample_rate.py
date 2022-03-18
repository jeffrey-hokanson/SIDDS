import numpy as np
from lide import *
from pgf import PGF
from scipy.spatial.distance import jaccard
from joblib import Memory, Parallel, delayed, parallel_backend
import socket
import _pickle as cPickle
import argparse
from scipy.interpolate import interp1d

from fig_noise import fit_lide, fit_lsopinf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_sample_rate', verbose=0)
else:
	memory = Memory('.cache', verbose=0)


def fit_lide(phis, Ys, dts, points = 3, oversample = 1, smooth = 1e-7, maxiter = 100, verbose = False, **kwargs):
	lide = LIDE(phis, Ys, dts, points = points, oversample = oversample, **kwargs)
	lide.solve(maxiter = maxiter, smooth = smooth)
	return lide.C


def sample_rate_study(fit, fname, x0, C, phis, dts = [1e-2], noise = 1e-3, M = 1000, Ntrials = 100, n_jobs = -1, **kwargs):
	
	jobs = []
	delayed_fit = delayed(memory.cache(fit, ignore = ['verbose']))
	cache_evolve_system = memory.cache(evolve_system)
	for dt in dts:
		t = dt * np.arange(M)
		X = cache_evolve_system(phis, C, x0, t)
		for seed in range(Ntrials):
			np.random.seed(seed)
			Y = X + noise*np.random.randn(*X.shape)
			jobs.append(delayed_fit(phis, [Y], [dt], **kwargs))
	
	print("running parallel job")
	with parallel_backend('loky', inner_max_num_threads = 1): 
		job_results = Parallel(n_jobs = n_jobs, verbose = 100, backend = 'loky')(jobs)
	print("running parallel job complete")

	idx = 0
	err = np.nan*np.zeros((len(dts), Ntrials))
	for i, dt in enumerate(dts):
		for j, seed in enumerate(range(Ntrials)):
			Cest = job_results[idx]
			idx += 1
			err[i,j] = np.linalg.norm(C - Cest, 'fro')
	
	pgf = PGF()
	pgf.add('dt', dts)
	p0, p25, p50, p75, p100 = np.nanpercentile(err, [0, 25, 50, 75, 100], axis = 1)
	pgf.add('F0', p0)
	pgf.add('F25', p25)
	pgf.add('F50', p50)
	pgf.add('F75', p75)
	pgf.add('F100', p100)
	pgf.write(fname)

if __name__ == '__main__':
	phis, C, x0 = vanderpol()
	x0 = np.array([ 1.23419042,-0.66946457]) # Start on the limit cycle
	dts = 10**(-2 + 0.05*np.arange(0,41))
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--oversample', type = int, default = 1)
	parser.add_argument('--points', type = int, default = 3)
	parser.add_argument('--trials', type = int, default = 100)
	parser.add_argument('--jobs', type = int, default = -1)
	args = parser.parse_args()


	Ntrials = args.trials
	n_jobs = args.jobs

	oversample = args.oversample
	points = args.points

	fname = f'data/fig_sample_rate_lide_points_{points:d}_oversample_{oversample:d}.dat'
	kwargs = {'oversample': oversample, 'points': points, 'verbose' : False, 'tol_dx' : 1e-10, 'tol_opt': 1e-8}
	sample_rate_study(fit_lide, fname, x0, C, phis, dts = dts, n_jobs = n_jobs, Ntrials = Ntrials, **kwargs)	

