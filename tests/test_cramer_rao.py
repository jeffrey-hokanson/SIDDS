import numpy as np
from lide import *
from lide.cramer_rao import *
import scipy.linalg

def test_cramer_rao():
	M = 10000
	dt = 1e-2
	#phis, C, x0 = vanderpol()
	phis, C, x0 = lorenz63()
	#phis, C, x0 = simple_harmonic_oscillator(include_constant = False)
	t = dt * np.arange(M)
	d, n = C.shape

	X = evolve_system(phis, C, x0, t)
	#mask = np.abs(C) > 1e-5
	mask = None
	cov_crb = cramer_rao(C, phis, [X], [dt], progress = True)
	print(cov_crb)
	I = np.eye( cov_crb.shape[0], d * n)
	C = (I.T @ (cov_crb @ I))
	print(C)
	

def test_tangent_space():
	M = 200
	dt = 1e-2
	#phis, C, x0 = lorenz63()
	phis, C, x0 = simple_harmonic_oscillator(include_constant = False)
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)

	Xs = [X]
	dts = [dt]

	mask = np.abs(C) > 1e-3

	U1 = construct_tangent_space(C, phis, Xs, dts, mask = mask)
	U2 = construct_tangent_space_fd(C, phis, Xs, dts, mask = mask)
	U3 = construct_tangent_space_one(C, phis, Xs, dts, mask = mask)

	ang = np.rad2deg(scipy.linalg.subspace_angles(U1, U2))
	print(ang)
	
	ang = np.rad2deg(scipy.linalg.subspace_angles(U1, U3))
	print(ang)


if __name__ == '__main__':
	test_cramer_rao()
	#test_cramer_rao_exact()
	#test_tangent_space()
