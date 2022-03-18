import numpy as np
import scipy.sparse
from generate_data import *
from pgf import PGF
from lide import *

import socket
from joblib import Memory, Parallel, delayed, parallel_backend
if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_noise', verbose=0)
else:
	memory = Memory('.cache', verbose=0)


def fit_lide(phis, Ys, dts, points = points):
	lide = LIDE(phis, Ys, dts, points = points, tol_opt = 1e-8, tol_dx = 1e-8) 
	lide.solve(maxiter = 100)
	return lide.C



if __name__ == '__main__':

	Ntrials = 100
	points_vec = np.arange(3,11, 2).astype(int)


	dts = [0.5e-2, 1e-2, 2e-2]
	for dt in dts:
		err_lide = np.nan * np.zeros((Ntrials, len(points_vec)))
		err_lsop = np.nan * np.zeros((Ntrials, len(points_vec)))
		for i, seed in enumerate(range(Ntrials)):
			print('='*10 + f"dt {dt} \t seed {seed}" + '='*10)
			Xs, Ys, phis, C, dts = generate_data(noise = 1e-4, M = int(10/dt), seed = seed, dt = dt)
			Sigmas = [scipy.sparse.eye(Ys[0].shape[0]*Ys[0].shape[1])]
			W = 0*scipy.sparse.eye(C.shape[0]*C.shape[1])
			for j, points in enumerate(points_vec):
				lide = LIDE(phis, Ys, Sigmas, W, dts, points, mu = 1e0, verbose = False)
				lide.solve(maxiter = 10)
				err_lide[i,j] = np.linalg.norm(lide.C - C, 'fro')

				C_lsop = lsopinf(Ys, phis, dts, points)
				err_lsop[i,j] = np.linalg.norm(C_lsop - C, 'fro')
				print(f'{points}\t 	{err_lide[i,j]:5e} \t {err_lsop[i,j]:5e}')
			
				pgf = PGF()
				pgf.add('points', points_vec)
				p0, p25, p50, p75, p100 = np.nanpercentile(err_lide, [0, 25, 50, 75, 100], axis = 0)
				pgf.add('p0', p0)
				pgf.add('p25', p25)
				pgf.add('p50', p50)
				pgf.add('p75', p75)
				pgf.add('p100', p100)
				pgf.write(f'data/fig_points_lide_{dt:5e}.dat')
				
				pgf = PGF()
				pgf.add('points', points_vec)
				p0, p25, p50, p75, p100 = np.nanpercentile(err_lsop, [0, 25, 50, 75, 100], axis = 0)
				pgf.add('p0', p0)
				pgf.add('p25', p25)
				pgf.add('p50', p50)
				pgf.add('p75', p75)
				pgf.add('p100', p100)
				pgf.write(f'data/fig_points_lsop_{dt:5e}.dat')
