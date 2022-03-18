import numpy as np
import scipy.sparse as ss
from lide import *


def estimate_median(cov_C, N = 100_000):
	L = np.linalg.cholesky(cov_C)
	Z = np.random.randn(cov_C.shape[1], N)
	LZ = L @ Z
	l2_norm = np.sqrt(np.sum(LZ**2, axis = 0))
	return np.percentile( l2_norm, [50])
	

# Duffing
phis, C, x0 = duffing()
dt = 1e-2
M = 1000
t = dt * np.arange(M)
X = evolve_system(phis, C, x0, t)

print("--------Duffing---------")
cov = cramer_rao(C, phis, [X], [dt])
P_C = ss.eye(cov.shape[0], C.shape[0] * C.shape[1]).A
cov_C = cov_C = P_C.T @ (cov @ P_C) 

print("sqrt trace", np.sqrt(np.trace(cov_C)))
print("median", estimate_median(cov_C))


print("--------Duffing Masked--------")
mask = np.abs(C) > 1e-5
cov = cramer_rao(C, phis, [X], [dt], mask = mask)
P_C = ss.eye(cov.shape[0], np.sum(mask)).A
cov_C = cov_C = P_C.T @ (cov @ P_C) 
print("sqrttrace", np.sqrt(np.trace(cov_C)))
print("median", estimate_median(cov_C))


# Lorenz63
phis, C, x0 = lorenz63()
dt = 1e-2
M = 2000
t = dt * np.arange(M)
X = evolve_system(phis, C, x0, t)

print("--------Lorenz63---------")
cov = cramer_rao(C, phis, [X], [dt])
P_C = ss.eye(cov.shape[0], C.shape[0] * C.shape[1]).A
cov_C = cov_C = P_C.T @ (cov @ P_C) 

print("sqrt trace", np.sqrt(np.trace(cov_C)))
print("median", estimate_median(cov_C))


print("--------Lorenz63 Masked--------")
mask = np.abs(C) > 1e-5
cov = cramer_rao(C, phis, [X], [dt], mask = mask)
P_C = ss.eye(cov.shape[0], np.sum(mask)).A
cov_C = cov_C = P_C.T @ (cov @ P_C) 
print("sqrttrace", np.sqrt(np.trace(cov_C)))
print("median", estimate_median(cov_C))


# Vanderpol
phis, C, x0 = vanderpol()
dt = 1e-2
M = 1000
t = dt * np.arange(M)
X = evolve_system(phis, C, x0, t)

print("--------Vanderpol---------")
cov = cramer_rao(C, phis, [X], [dt])
P_C = ss.eye(cov.shape[0], C.shape[0] * C.shape[1]).A
cov_C = cov_C = P_C.T @ (cov @ P_C) 

print("sqrt trace", np.sqrt(np.trace(cov_C)))
print("median", estimate_median(cov_C))


print("--------Vanderpol Masked--------")
mask = np.abs(C) > 1e-5
cov = cramer_rao(C, phis, [X], [dt], mask = mask)
P_C = ss.eye(cov.shape[0], np.sum(mask)).A
cov_C = cov_C = P_C.T @ (cov @ P_C) 
print("sqrttrace", np.sqrt(np.trace(cov_C)))
print("median", estimate_median(cov_C))

