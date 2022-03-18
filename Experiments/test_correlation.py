import numpy as np
import scipy.sparse
from lide import *
M = 10

cor = 0	
Sigma = np.array([[1, cor],[cor, 1]])
transform = np.linalg.cholesky(Sigma)
print(transform)
Ntrial = int(1e7)
noises = np.zeros((Ntrial, 2*M))
for i in range(Ntrial):
	noises[i,:] = vec(transform @ np.random.randn(2, M))

C = np.cov(noises.T)
A = scipy.sparse.kron(transform, scipy.sparse.eye(M))
Sigma = A @ A.T
print(np.linalg.norm(C - Sigma.A,'fro'))
