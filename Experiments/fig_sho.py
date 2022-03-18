import numpy as np
from lide import *
import socket
from pgf import PGF
import scipy.sparse
import matplotlib.pyplot as plt

from joblib import Memory
from lide.modified_sindy import *

if 'colorado' in socket.gethostname():
	memory = Memory('/rc_scratch/jeho8774/lide_fig_noise', verbose=0)
else:
	memory = Memory('.cache', verbose=0)

cache_evolve_system = memory.cache(evolve_system)

if __name__ == '__main__':
	dt = 1e-2
	M = 2000	
	phis, C, x0 = simple_harmonic_oscillator(include_constant = False)
	dim = phis[0].dim
	t = dt * np.arange(M)
	X = cache_evolve_system(phis, C, x0, t)

	Y = np.copy(X)
	np.random.seed(0)
	Y += np.random.randn(*X.shape)

	oversample = 1	
	for phi, c in zip(phis, C.T):
		print(phi.degree, c)


	lide = LIDE(phis, [Y], [dt], verbose = True, points = 3)
	lide.solve()

	#lide.solve_irls(alpha = 0.5, q=0)

	print("True")
	print(C[:,::-1])
	print("Recovered")
	print(lide.C[:,::-1])
	Cest = lsopinf(phis, [Y], [dt], points = 3)
	print("LS\n", Cest[:,::-1])
	
	fig, axes = plt.subplots(2,2)
	for ax, x, y in zip(axes[:,0], Y, lide.Xs[0]):
		ax.plot(t, x,'.')
		ax.plot(t, y)

	axes[1,1].plot(Y[0], Y[1],'.')	
	axes[1,1].plot(lide.Xs[0][0], lide.Xs[0][1])	

	pgf = PGF()
	pgf.add('t', t)
	pgf.add('x1', X[0])	
	pgf.add('x2', X[1])	
	pgf.add('y1', Y[0])	
	pgf.add('y2', Y[1])	
	pgf.add('z1', lide.Xs[0][0])	
	pgf.add('z2', lide.Xs[0][1])	
	pgf.write(f'data/fig_sho.dat')

	# Now with the sparsity structure given
	mask = np.abs(C) > 0

	print("=========With fixed sparsity=========")	
	C = lsopinf(phis, [Y], [dt], mask = mask)
	print("LSOI")
	print(C[:,::-1])

	lide = LIDE(phis, [Y], [dt], verbose = True, points = 3, mask = mask)
	lide.solve()
	print("LIDE")
	print(lide.C[:,::-1])
	
	print("mSINDy")
	C, Xs = modified_SINDy(phis, [Y], [dt], lam = 0.1)
	print(C[:,::-1])

	print("LIDE with sparsity")
	lide = LIDE(phis, [Y], [dt], verbose = True, points = 3)
	lide.solve_irls(alpha=1, q = 0)
	print(lide.C[:,::-1])
	print("polish")
	lide.solve_polish()
	print(lide.C[:,::-1])
	
