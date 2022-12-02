import numpy as np
import lide

from pgf import PGF

dt = 1e-2
N = 10/dt

phis, C = lide.vanderpol()
x0 = np.array([1,0])
t = dt * np.arange(N)

X = lide.evolve_system(phis, C, x0, t)

pgf = PGF()
pgf.add('x', X[0,:])
pgf.add('y', X[1,:])
pgf.write('data/fig_vanderpol.dat')


np.random.seed(0)
N = np.random.randn(*X.shape)
pgf = PGF()
pgf.add('x', (X+N)[0,:])
pgf.add('y', (X+N)[1,:])
pgf.write('data/fig_vanderpol_noise.dat')


