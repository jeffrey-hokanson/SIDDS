import numpy as np
from lide import *



dt = 1e-2
M = 1000
phis, C, x0 = duffing()
t = dt * np.arange(M)
X = evolve_system(phis, C, x0, t)
var = np.mean(np.var(X, axis = 1))
print("Duffing")
print(var)

dt = 1e-2
M = 2000
phis, C, x0 = lorenz63()
t = dt * np.arange(M)
X = evolve_system(phis, C, x0, t)
var = np.mean(np.var(X, axis = 1))
print("Lorenz63")
print(var)

dt = 1e-2
M = 1000
phis, C, x0 = vanderpol()
t = dt * np.arange(M)
X = evolve_system(phis, C, x0, t)
var = np.mean(np.var(X, axis = 1))
print("Vanderpol")
print(var)
