import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from .pareto import pareto_corner 

def finite_diff_matrix(n, deriv = 1, points = 9):
	# Generic coefficients
	bw = (points - 1) //2
	x = np.arange(-bw, bw+1, dtype = float)
	V = np.vstack([x**p for p in range(points)])
	b = np.zeros(points)
	b[deriv] = np.math.factorial(deriv)
	lu = scipy.linalg.lu_factor(V)	
	w = scipy.linalg.lu_solve(lu, b)
	#w += scipy.linalg.lu_solve(lu, b - V @ w) # One step of iterative refinement

	offsets = np.arange(-bw, bw+1, dtype = int)
	diags = [w[i]*np.ones(n - k) for i, k in enumerate(offsets)]
	D = scipy.sparse.diags(diags, offsets, shape = (n, n)).todok()
	
	# Fix beginning/ending points
	for i in range(bw):
		x = np.arange(-i, points - i, dtype = float)
		V = np.vstack([x**p for p in range(len(x))])
		b = np.zeros(len(x))
		b[deriv] = np.math.factorial(deriv)
		lu = scipy.linalg.lu_factor(V)	
		w = scipy.linalg.lu_solve(lu, b)
		#w += scipy.linalg.lu_solve(lu, b - V @ w) # One step of iterative refinement
		for j in range(len(w)):
			D[i, j] = w[j]
			D[-i-1,-j-1] = (-1)**deriv * w[j]

	return D.tocsr()


def denoise_fixed(Y, dt, lam = 1e-2, deriv = 2, points = 3):
	n = Y.shape[1]
	Yrec = np.zeros_like(Y)
	D = finite_diff_matrix(n, deriv = deriv, points = points) / dt**deriv

	A = scipy.sparse.eye(n) + lam * D.T @ D
	lu = scipy.sparse.linalg.splu(A.tocsc())
	for i, y in enumerate(Y):
		Yrec[i,:] = lu.solve(y)
	return Yrec

def denoise_corner(Y, dt, lam_min = 1e-5, lam_max = 1e2, deriv = 2, points = 3):
	n = Y.shape[1]
	Yrec = np.zeros_like(Y)
	D = finite_diff_matrix(n, deriv = deriv, points = points) / dt**deriv
	
	def fun(lam, **kwargs):
		A = scipy.sparse.eye(n) + lam * D.T @ D
		lu = scipy.sparse.linalg.splu(A.tocsc())
		for i, y in enumerate(Y):
			Yrec[i,:] = lu.solve(y)
		obj = np.linalg.norm(Yrec - Y,'fro')**2
		reg = np.linalg.norm(Yrec @ D.T, 'fro')**2

		return {'obj': obj, 'reg': reg, 'kwargs': {'Yrec': Yrec}}

	lam2, lam3, ret2, ret3 = pareto_corner(fun, lam_min, lam_max, verbose = True)
	return ret2['kwargs']['Yrec']
