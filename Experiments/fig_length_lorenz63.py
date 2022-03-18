import numpy as np
import lide
from pgf import PGF
from functools import partial


algs = [
	lide.lsopinf, 
	lambda *args, **kwargs: lide.lide(*args, **kwargs)[0],
]
names = [
	'lsopinf', 
	'lide',
]

# Regularized LIDE
def first(f):
	return lambda *args, **kwargs: f(*args, **kwargs)[0]


for p in [0,1]:	
	for lam in [ 1e-10,1e-8, 1e-6,1e-4,1e-2]:
		algs += [first(partial(lide.lide_regularize,p=p,lam = lam, maxiter = 100))]
		names += [f'lide+ell{p:1d}+lam={lam:.0e}']	

# Regularized LSOPINF
for p in [0,1]:	
	for lam in [1e-10,1e-8, 1e-6,1e-4,1e-2]:
		algs += [partial(lide.lsopinf_regularize, p=int(p), lam = float(lam))]
		names += [f'lopinf+ell{p:1d}+lam={lam:.0e}']	


def fig_length_lorenz63(Ntrial = 10, N = 220, points = 9, dt = 1e-2, noise = 1e-3, lengths = None):
	phis, C = lide.lorenz63()
	x0 = np.array([-8, 7,27])
	t = dt * np.arange(N)
	X = lide.evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = [dt]
	
	errors = np.nan*np.zeros((len(lengths), Ntrial, len(algs)))

	for i, length in enumerate(lengths):
		for j, seed in enumerate(range(Ntrial)):
			np.random.seed(seed)
			Ys = [X + noise * np.random.randn(*X.shape) for X in Xs]
			Ys = [Y[:length] for Y in Ys]
			for k, alg in enumerate(algs):
				try:
					Cest = alg(Ys, phis, dts, points = points, C_true = C, verbose = 2)
					errors[i,j,k] = np.linalg.norm(C - Cest, 'fro')
				except KeyboardInterrupt:
					raise KeyboardInterrupt
				except:
					pass
				print(f'length {length:5d}, seed {seed:2d}, alg {names[k]:25s}, error {errors[i,j,k]:10e}')
				
				# Update the percentiles

				percs = np.nan*np.zeros((len(lengths), 5))
				for ii, _ in enumerate(lengths):
					percs[ii,:] = np.nanpercentile(errors[ii,:,k], [0,25,50, 75, 100])
				pgf = PGF()
				pgf.add('length', lengths)
				pgf.add('p0', percs[:,0])
				pgf.add('p25', percs[:,1])
				pgf.add('p50', percs[:,2])
				pgf.add('p75', percs[:,3])
				pgf.add('p100', percs[:,4])
				pgf.write(f'data/fig_length_lorenz63_{names[k]}.dat')

if __name__ == '__main__':
	# Alex uses N=220, dt=1e-2
	# SINDy (BPK16) uses N=100/dt, dt=1e-3 (p.17)
	#algs = algs[0:2]
	#names = names[0:2]
	dt = 1e-2
	lengths = np.unique(np.logspace(2, np.log10(100/dt),30 +1).astype(int))
	fig_length_lorenz63(Ntrial = 20, N = 100/dt, dt = dt, points = 9, lengths = lengths)
	
