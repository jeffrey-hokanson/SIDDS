import numpy as np
import lide

from util import *

def fd_derivative(fun, x, h = 2e-7):
	f = fun(x)
	dx = np.zeros(list(f.shape) + [ len(x)], dtype = float)
	
	for i in range(len(x)):
		ei = np.zeros_like(x)
		ei[i]= 1.		
		f1 = fun(x + ei * h)
		f2 = fun(x - ei * h)
		#print(f1, f2, (f1-f2)/(2*h))
		dx[...,i] = (f1 - f2)/(2*h)
		#print(dx[...,i])

	return dx

def test_lsopinf():
	
	Ys, phis, dts, C = generate_data(noise = 0, )

	Cest = lide.lsopinf(Ys, phis, dts)
	print(Cest)
	print(C)
	print("C error", np.linalg.norm(C - Cest,'fro'))

def test_lsopinf_regularize():
	
	Ys, phis, dts, C = generate_data(noise = 1e-5)

	Cest = lide.lsopinf_regularize(Ys, phis, dts, C_true = C, lam = 1e-5, p = 0, points = 9)

	print("C error", np.linalg.norm(C - Cest,'fro'))


def test_lsopinf_grad(mask = True):
	M = 20
	dt = 1e-2
	#phis, C, x0 = vanderpol()
	#phis, C, x0 = lorenz63()
	t = dt * np.arange(M)
	phis, C, x0 = lide.simple_harmonic_oscillator(include_constant = False)
	X = lide.evolve_system(phis, C, x0, t)
	d, n = C.shape

	if mask:
		mask = np.abs(C) > 1e-3
	else:
		mask = None

	A = lide.lsopinf_grad(phis, [X], [dt], points = 3, mask = mask)

	def lsopinf_flat(vec_x, points = 3):
		
		C = lide.lsopinf(phis, [lide.unvec(vec_x, d)], [dt], points = points, mask = mask)
		vec_c = lide.vec(C)
		if mask is None:
			return vec_c
		else:
			return vec_c[lide.vec(mask)]

	A_est = fd_derivative(lsopinf_flat, lide.vec(X), h = 1e-7)
	print("A_est", A_est.shape)
	print("A", A.shape)
	print("A")
	print(A.T)
	print("A est")
	print(A_est.T)
	print("ratio")
	print( (A/A_est).T)
	
	assert np.allclose(A, A_est, rtol = 1e-5)

	print("Cov")
	print(A @ A.T)

if __name__ == '__main__':
	#test_lsopinf_regularize()
	#test_lsopinf()
	test_lsopinf_grad()
