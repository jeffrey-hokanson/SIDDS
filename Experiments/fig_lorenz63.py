import numpy as np
import lide

import matplotlib.pyplot as plt
from pgf import PGF

dt = 1e-2
N = 10/dt

phis, C = lide.lorenz63()
x0 = np.array([-8, 7,27])

x1 = np.array([-8+8e-2, 7,27])
t = dt * np.arange(N)
X1 = lide.evolve_system(phis, C, x0, t)
X2 = lide.evolve_system(phis, C, x1, t)


pgf = PGF()
pgf.add('x', X1[0,:])
pgf.add('y', X1[1,:])
pgf.add('z', X1[2,:])
pgf.write('data/fig_lorenz63a.dat')

pgf = PGF()
pgf.add('x', X2[0,:])
pgf.add('y', X2[1,:])
pgf.add('z', X2[2,:])
pgf.write('data/fig_lorenz63b.dat')


X = X1
np.random.seed(0)
N = np.random.randn(*X.shape)

pgf = PGF()
pgf.add('x', (X+N)[0,:])
pgf.add('y', (X+N)[1,:])
pgf.add('z', (X+N)[2,:])
pgf.write('data/fig_lorenz63_noise.dat')


