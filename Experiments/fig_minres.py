import numpy as np
from scipy.sparse.linalg import minres
from lide import *
from lide import vanderpol
from generate_data import *
from pgf import PGF

def solve_kkt(A, b, M = None, x0 = None, xt = None):
	res = []
	dist = []
	def callback(x):
		res.append(np.linalg.norm(A @ x - b))
		if xt is not None:
			dist.append(np.linalg.norm(x - xt))
		
	x, info = minres(A, b, x0 = x0, M = M, callback = callback, tol = 1e-20, maxiter =2000)

	if xt is not None:
		return x, np.array(res)/np.linalg.norm(b), np.array(dist)
	else:
		return x, np.array(res)/np.linalg.norm(b)


def kkt_experiment(M = 500, mu = 0.1, points = 9, precondition = False, noise = 1e-2, xt = None, warmstart = True):
	np.random.seed(0)
	Xs, Ys, phis, C, dts = generate_data(vanderpol, M = M, noise = noise)
	Ct = C + noise*np.random.randn(*C.shape)
	
	lide = LIDE(phis, Ys, dts, mu = mu, points = points, verbose = False)
	lide._obj.mu = 0
	lide.solve(maxiter = 1)

	x = lide.encode(lide.C, lide.Xs) 
	h = lide.constraints.fun(x)
	A = lide.constraints.jac(x)
	z = lide.z
	g = lide._obj.jac(x)
	B = lide._obj.hess(x)
	dp = lide.solve_relaxation(x, h, A)
	c = A @ dp
	AA, bb = lide.build_kkt_system(x, z, g, h, c, A, B)	
	if precondition:
		MM = lide.build_kkt_preconditioner(x, z, g, h, c, A, B)
	else:
		MM = None 
	if warmstart:
		x0 = np.hstack([dp, z])
	else:
		x0 = None
	return solve_kkt(AA, bb, M = MM, x0 = x0, xt = xt)
			

def fig_minres_converge(**kwargs):
	xt, _ = kkt_experiment(precondition = True, **kwargs)
	
	_, res, dist = kkt_experiment(xt = xt, **kwargs) 
	pgf = PGF()
	pgf.add('res', res)
	pgf.add('dist', dist/np.linalg.norm(xt))
	pgf.write(f'data/fig_minres_converge_M_{kwargs["M"]:d}_mu_{kwargs["mu"]:e}_warm.dat')
	
	_, res, dist = kkt_experiment(xt = xt, precondition = True, **kwargs) 
	pgf = PGF()
	pgf.add('res', res)
	pgf.add('dist', dist/np.linalg.norm(xt))
	pgf.write(f'data/fig_minres_converge_pre_M_{kwargs["M"]:d}_mu_{kwargs["mu"]:e}_warm.dat')
	

	_, res, dist = kkt_experiment(xt = xt, warmstart = False, **kwargs) 
	pgf = PGF()
	pgf.add('res', res)
	pgf.add('dist', dist/np.linalg.norm(xt))
	pgf.write(f'data/fig_minres_converge_M_{kwargs["M"]:d}_mu_{kwargs["mu"]:e}.dat')
	
	_, res, dist = kkt_experiment(xt = xt, precondition = True, warmstart = False, **kwargs) 
	pgf = PGF()
	pgf.add('res', res)
	pgf.add('dist', dist/np.linalg.norm(xt))
	pgf.write(f'data/fig_minres_converge_pre_M_{kwargs["M"]:d}_mu_{kwargs["mu"]:e}.dat')

def fig_minres_mu(mus, M = 4000, tols = [1e-5]):
	
	nits = [[] for t in tols]
	for j, mu in enumerate(mus):
		xt, _ = kkt_experiment(precondition = True, mu = mu, M = M)
		_, res, dist = kkt_experiment(xt = xt, M = M, precondition = True, mu = mu) 
		for k, t in enumerate(tols):
			nits[k].append(np.argmax(dist <= t))
		print(mu, [nit[-1] for nit in nits])
		pgf = PGF()
		pgf.add('mu', mus[:j+1])
		for k, t in enumerate(tols):
			pgf.add(f'it_{t:5e}',nits[k])
		pgf.write(f'data/fig_minres_mu_M_{M:d}.dat')


if __name__ == '__main__':
	if True:
		fig_minres_converge(M = 4000, mu = 1e2)
	
	if False:
		for M in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
			print(M)
			kkt_experiment(prefix = 'data/fig_minres', M = M)

	if False:
		for M in [1000, 2000, 4000, 8000, 16000]:
			fig_minres_mu(np.logspace(-5,5, 10*3+1), M = M, tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])


#	import matplotlib.pyplot as plt
#	fig, ax = plt.subplots()
#	ax.set_yscale('log')
#	
#	M = 1000
#	mu = 1e0
#	points = 9
#	for mu in [1e-2,1e-1, 1e0, 1e1, 1e2]:
#		xt, res = kkt_experiment(M = M, mu = mu, points = points, precondition = True)
#		_, res, dist = kkt_experiment(M = M, mu = mu, points = points, precondition = True, xt = xt)
#		ax.plot(dist/np.linalg.norm(xt))
#	plt.show()
		
#	fig, ax = plt.subplots()
#
#	for mu in 10.**(np.linspace(-5,5,4*(5+5)+1)):
#		res = kkt_experiment(M = 100, mu = mu, points = 9, precondition = True, noise = 1e-5)
#		ax.plot(res)
#		try: 
#			print(f'{mu:5e}\t {np.min(np.argwhere(res <= 1e-9)):d}')
#		except ValueError:
#			print(f'{mu:5e}\t None')
#			
#	ax.set_yscale('log')
#	plt.show()
	
