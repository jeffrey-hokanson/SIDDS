import numpy as np
import lide
from lide.matrices import *
import scipy.sparse
from itertools import product
from lide.basis import *

M = 1000
d = 5
points = 5
phis = []
max_degree = 2	
for k, degree in enumerate(product(range(max_degree+1), repeat =d)):
	phis.append(Monomial(degree))

C = np.random.randn(d, len(phis))
X = np.random.randn(d, 1000)
D, M = make_DM(1000, points)
Phi = make_Phi( X, phis)
vec_grad_Phi = make_grad_vec_Phi(X, phis, C)
vec_Phi = make_vec(Phi, d)
vec_M = make_vec(M, d)
vec_D = make_vec(D, d)

Ax = vec_D - vec_M @ vec_grad_Phi
Ax = scipy.sparse.coo_matrix(Ax)

B = scipy.sparse.coo_matrix(Ax.T @ Ax)
bandwidth = 0
for i, j, v in zip(B.row, B.col, B.data):
	if np.abs(v) > 0:
		bandwidth = max(bandwidth, abs(i-j))

print(bandwidth, (points - 1)*d)
#print(2*bandwidth+1)
