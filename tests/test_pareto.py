import numpy as np
import scipy.linalg
import scipy.sparse
from lide import *

def fig_pareto():
	np.random.seed(0)
	phis, C, x0 = lorenz63()
	print("true C")
	print(C)
	dt = 5e-2
	M = 500
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])

	Ys = [X + 1e-4*np.random.randn(*X.shape) for X in Xs]
	q = 0
	C0 = np.zeros(C.shape)
	Xs0 = [0*Y for Y in Ys]
	def fun(lam):
		lide = LIDE(phis, Ys, dts, points = 9, verbose = True, mu = 1e-3, tol_dx = 1e-10, tol_opt = 1e-6, v_soc = -1)
		lide.solve_irls(alpha = lam, q = q, epsilon = 1e-3, maxiter = 500)
		Xs = lide.Xs
		C = lide.C
		obj = lide.objective()
		reg = lide.regularization()
		
		w = np.power(np.abs( vec(np.abs(lide.C))**2 + lide.epsilon), q/2. - 1)
		reg_act = vec(C).T @ scipy.sparse.diags(w) @ vec(C)
		return {'obj': obj, 'reg': reg, 'reg_act': reg_act, 'C': np.copy(C)}

	if False:
		ret = fun(1e-2)
		print(ret)
		assert False
	
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.set_xlabel('objective')
	ax.set_ylabel('regularizer')
	ax.set_yscale('log')
	ax.set_xscale('log')
	for lam in np.logspace(-2, 1, 20):
		ret = fun(lam)
		print(lam, ret)
		ax.plot(ret['obj'],ret['reg'],'b.')
		ax.plot(ret['obj'],ret['reg_act'],'r.')
	plt.show()	


if __name__ == '__main__':
	fig_pareto()
