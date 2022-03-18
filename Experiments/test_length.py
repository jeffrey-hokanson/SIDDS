
import numpy as np
from lide.examples import lorenz63
from lide import evolve_system
from pgf import PGF
from functools import partial

from fig_noise import *

cache_evolve = memory.cache(evolve_system)
points = 9
dt = 1e-2
noise = 1e-2
M = int(1e6)
phis, C, x0 = lorenz63()
dim = phis[0].dim

t = dt * np.arange(M)
cache_evolve = memory.cache(evolve_system)

print("evolving system")
X = cache_evolve_system(phis, C, x0, t)
print("done evolving")
Xs = [X[:,:int(8e5)]]
Ys = [X + noise * np.random.randn(*X.shape) for X in Xs]
dts = [dt]

fit_lide(Ys, phis, dts, points, solve_qp_kwargs = {'verbose': True})	

