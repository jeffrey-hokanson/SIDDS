import numpy as np
from lide import *
from scipy.optimize import minimize_scalar

phis, C, x0 = vanderpol()
x0 = np.array([ 1.23419042,-0.66946457]) # Start on the limit cycle

def mismatch(t):
	t = float(t)
	x = evolve_system(phis, C, x0, np.array([0,t]))
	return np.linalg.norm(x[:,1] - x0)

if False:
	# This points to a minimizer in [7.4, 8]
	t = np.linspace(0, 10, 100)
	for ti in t[1:]:
		print(ti, mismatch(ti))

res = minimize_scalar(mismatch, [7.4, 8])
print(res)
