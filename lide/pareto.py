# Based on the paper: https://doi.org/10.1088/2633-1357/abad0d
import numpy as np
from iterprinter import IterationPrinter

def menger(P1, P2, P3):
	r""" Compute the curvature of a circle from three points 
	"""

	x1, y1 = P1
	x2, y2 = P2
	x3, y3 = P3

	P1P2 = (x2 - x1)**2 + (y2 - y1)**2
	P2P3 = (x3 - x2)**2 + (y3 - y2)**2
	P3P1 = (x1 - x3)**2 + (y1 - y3)**2

	if P1P2 * P2P3 * P3P1 == 0:
		return 0

	C2 = 2*( x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)/(
			np.sqrt(P1P2*P2P3*P3P1))

	return C2


def golden_section_search(fun, a, b, tol = 1e-10, maxiter = 100):
	r""" Golden section search for a minimizer on the interval [a, b]

	This implementation follows that on the Wikipedia
	https://en.wikipedia.org/wiki/Golden-section_search

	Parameters
	----------
	fun: function
		A single parameter, scalar-valued function
	a: float
		left endpoint
	b: float
		right endpoint
	"""
	phi = (1+np.sqrt(5))/2.

	it = 0
	while abs(b - a) > tol and it < maxiter:
		c = b - (b - a)/phi
		d = a + (b - a)/phi
		if fun(c) < fun(d):
			b = d
		else:
			a = c

		it += 1

	return (a + b)/2

def pareto_corner(fun, lam_min, lam_max, tol = 1e-2, maxiter = 20, kwargs = {}, verbose = True):
	r""" Find the point of greatest curvature on the Pareto frontier 

	fun: function
		Of the format  {'obj': float, 'reg': float, 'kwargs': dict} = fun(lam, **kwargs)
		where 'obj' refers to the objective value and 'reg' refers to the regularization
		and input 'lam' corresponds to the regularization parameter.
		The additional kwargs are passed to the adjacent function call 
	"""
	
	if verbose:
		printer = IterationPrinter(it = '4d',lam1 = '8.2e', lam2 = '8.2e', lam3 = '8.2e', lam4 = '8.2e', C2 = '11.4e', C3 = '11.4e',
			obj2 = '8.2e', reg2 = '8.2e', obj3 = '8.2e', reg3 = '8.2e')
		printer.print_header(it = 'iter', lam1 = 'lam1', lam2 = 'lam2', lam3 = 'lam3', lam4 = 'lam4', C2 = 'C2', C3 = 'C3',
			obj2 = 'obj2', reg2 = 'reg2', obj3 = 'obj3', reg3 = 'reg3')

	phi = (1 + np.sqrt(5))/2.

	log_lam1, log_lam4 = np.log10(lam_min), np.log10(lam_max)

	ret1 = fun(10.**log_lam1, **kwargs)
	ret4 = fun(10.**log_lam4, **kwargs)
	
	log_lam2 = log_lam4 - (log_lam4 - log_lam1)/phi
	log_lam3 = log_lam1 + (log_lam4 - log_lam1)/phi

	ret2 = fun(10.**log_lam2, **ret1['kwargs'])
	ret3 = fun(10.**log_lam3, **ret4['kwargs'])

	it = 0 
	while not np.isclose(10**log_lam2, 10**log_lam3, atol = 0, rtol = tol) and maxiter > it:
		assert log_lam1 < log_lam2 and log_lam2 < log_lam3 and log_lam3 < log_lam4

		# Compute points on the pareto frontier
		P1 = [np.log(ret1['obj']), np.log(ret1['reg'])]
		P2 = [np.log(ret2['obj']), np.log(ret2['reg'])]
		P3 = [np.log(ret3['obj']), np.log(ret3['reg'])]
		P4 = [np.log(ret4['obj']), np.log(ret4['reg'])]

		#line = ''
		#for P in [P1,P2,P3,P4]:
		#	line+= f'({P[0]:5.2f}, {P[1]:5.2f}) '
		#print(line)
		C2 = menger(P1, P2, P3)
		C3 = menger(P2, P3, P4)
		if verbose:
			printer.print_iter(it = it, lam1 = 10**log_lam1, lam2 = 10**log_lam2, lam3 = 10**log_lam3, lam4 = 10**log_lam4, C2= C2, C3 = C3,
			obj2 = ret2['obj'], reg2 = ret2['reg'], obj3 = ret3['obj'], reg3 = ret3['reg'])
		if C3 < 0 or (C2 > C3):
			# If we haven't bracketed the the root
			log_lam4, log_lam3 = log_lam3, log_lam2
			ret4, ret3 = ret3, ret2
			log_lam2 = log_lam4 - (log_lam4 - log_lam1)/phi
			ret2 = fun(10.**log_lam2, **ret1['kwargs'])
		else:
			log_lam1, log_lam2 = log_lam2, log_lam3
			ret1, ret2 = ret2, ret3
			log_lam3 = log_lam1 + (log_lam4 - log_lam1)/phi
			ret3 = fun(10.**log_lam3, **ret4['kwargs'])
		
		it += 1
	lam2 = 10**log_lam2
	lam3 = 10**log_lam3
	return lam2, lam3, ret2, ret3

class TestProblem:
	def __init__(self, A, b):
		self.A = A
		self.b = b
	def __call__(self, lam, **kwargs):
		AA = np.vstack([self.A, lam*np.eye(self.A.shape[1]) ])
		bb = np.hstack([self.b, np.zeros(self.A.shape[1])] )
		x = np.linalg.lstsq(AA, bb, rcond = None)[0]
		
		ret = {'obj': np.linalg.norm(A @ x - b), 'reg': np.linalg.norm(x), 'kwargs': {} }
		return ret




if __name__ == '__main__':

	if False:
		np.random.seed(0)
		M, N = 100, 50
		A = np.random.randn(M, N)
		# Make this problem ill-conditioned
		U, s, VT = np.linalg.svd(A, full_matrices = False)
		A = U @ np.diag(s * np.exp(-0.5*np.arange(len(s))))@ VT
		x = np.random.randn(N)
		b = A @ x + 1e-5*np.random.randn(M)
		fun = TestProblem(A, b)

	if True:
		from lide2 import *
		from examples import vanderpol
		phis, C = vanderpol()
		x0 = np.array([0,1])
		t = dt * np.arange(M)
		X = evolve_system(phis, C, x0, t)
		Xs = [X]
		dts = np.array([dt])

		Ys = [X + 1e-2*np.random.randn(*X.shape) for X in Xs]
		def fun(lam):
			lide = LIDE(phis, Ys, dts, points = 9, verbose = True)
			lide.solve_irls(rho = lam)
			Xs = lide.Xs
			obj = sum([np.linalg.norm(Y - X,'fro')**2 for Y, X, in zip(Ys, Xs)])
			w = np.power(np.abs( vec(np.abs(lide.C))**2 + lide.epsilon), q/2. - 1)
			reg = vec(C).T @ scipy.sparse.diags(w) @ vec(C)
			return {'obj': obj, 'reg': reg}
			
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.set_xlabel('objective')
	ax.set_ylabel('regularizer')
	ax.set_yscale('log')
	ax.set_xscale('log')
	for lam in np.logspace(-5, 2, 20):
		ret = fun(lam)
		ax.plot(ret['obj'],ret['reg'],'b.')
	plt.show()	

	pareto_corner(fun, 1e-15, 1e2)

