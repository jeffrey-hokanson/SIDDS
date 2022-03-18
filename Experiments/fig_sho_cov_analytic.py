import numpy as np
from lide import *
import socket
from pgf import PGF
import scipy.sparse
import matplotlib.pyplot as plt

def write_matrix(fname, mat):
	with open(fname, 'w') as f:
		for mat_row in mat:
			f.write(','.join([f'{c:18.10e}' for c in mat_row]) + '\n' )


if __name__ == '__main__':
	phis, C, x0 = simple_harmonic_oscillator(include_constant = False)
	mask = np.abs(C)> 1e-4
	#mask = np.ones(C.shape, dtype = bool)
	#mask = None
	d = phis[0].dim
	dt = 1e-2	
	t = dt * np.arange(2000)
	X = evolve_system(phis, C, x0, t)
	d, m = X.shape

	CRB = cramer_rao(C, phis, [X], [dt], mask = mask )	
	P_C = scipy.sparse.eye(CRB.shape[0], np.sum(mask)).A
	cov_crb = P_C.T @ (CRB @ P_C)

	write_matrix('data/fig_sho_cov_analytic_crb.csv', cov_crb)
	print(cov_crb)

	# LSOI estimate
	for points in [3,5,7, 9]:
		A = lsopinf_grad(phis, [X], [dt], points = points, mask = mask)
		cov_lsoi = A @ A.T
		print("LSOI points =", points)
		print(cov_lsoi)
		write_matrix(f'data/fig_sho_cov_analytic_lsoi_pts{points}.csv', cov_lsoi)
		print()
	
	for points in [3,5,7, 9]:
		A = lsopinf_grad(phis, [X], [dt], points = points, mask = mask, square = False)
		cov_lsoi = A @ A.T
		print("LSOI points =", points)
		print(cov_lsoi)
		write_matrix(f'data/fig_sho_cov_analytic_lsoi_rect_pts{points}.csv', cov_lsoi)
		print()

	# SIDDS
	for points in [3, 5, 7, 9]:
		cov_sidds = lide_covariance_C(C, phis, [X], [dt], mask = mask, points = points) 
		print("SIDDS points =", points)
		print(cov_sidds)
		print("D-efficiency", np.linalg.det(cov_crb)/np.linalg.det(cov_sidds))
		write_matrix(f'data/fig_sho_cov_analytic_lide_pts{points}.csv', cov_sidds)
		bias_sidds = lide_bias_C(C, phis, [X], [dt], mask = mask, points = points)
		print("SIDDS bias", bias_sidds, np.linalg.norm(bias_sidds))
		write_matrix(f'data/fig_sho_cov_analytic_lide_pts{points}_bias.csv', bias_sidds.reshape(-1,1))
		print()
