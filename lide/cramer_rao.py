import numpy as np
from .lide import LIDENonlinearEqualityConstraint
import scipy.linalg as sl
import scipy.sparse as ss 
from scipy.linalg import null_space, inv
import scipy.sparse
from scipy.sparse import block_diag, eye, diags, bmat, vstack
from .sensitivity import sensitivity
from .sidds_nullspace import Constraint

class LowRankSymmetric(ss.linalg.LinearOperator):
	r"""

	A = U @ M @ U.T
	"""
	def __init__(self, M, U):
		self.M = M
		self.U = U
	
	def _matmat(self, X):
		return self.U @ (self.M @ (self.U.T @ X))

	
	@property
	def shape(self):
		return self.U.shape[0], self.U.shape[0]

	@property
	def dtype(self):
		return self.M.dtype

def cramer_rao(C, phis, Xs, dts, Ms = None, mask = None, tangent = 'one', **kwargs):
	r"""
	Parameters
	----------
	Ms:
		inverse covariance matrices
	"""
	d, m = C.shape
	if Ms is None:
		Ms = [ss.eye(X.shape[0]*X.shape[1]) for X in Xs]

	if mask is None:
		mask = np.ones(C.shape, dtype = bool)
	
	if tangent == 'one':
		U = construct_tangent_space_one(C, phis, Xs, dts, mask = mask, **kwargs)
	else:
		U = construct_tangent_space(C, phis, Xs, dts, mask = mask, **kwargs)

	# Information matrix
	J = ss.block_diag([ss.diags(np.zeros(np.sum(mask))) ] + Ms)
	UJU = (U.T @ J @ U)
	V, s, VT = sl.svd(UJU)
	inv_UJU = V @ np.diag(1./s) @ V.T
	return LowRankSymmetric(inv_UJU, U)


def construct_tangent_space_fd(C, phis, Xs, dts, mask = None, points = 9):
	# Gradient of constraints
	constraint = LIDENonlinearEqualityConstraint(phis, Xs, dts, points, mask = mask)
	F = constraint.jac(constraint.encode(C, Xs))
	U = sl.null_space(F.A)

	return U

def construct_tangent_space(C, phis, Xs, dts, mask = None, **kwargs):
	# Gradient of constraints
	if len(Xs) > 1:
		raise NotImplementedError


	if mask is None:
		mask = np.ones(C.shape, dtype = bool)

	vec_mask = mask.flatten('F')	

	d, n = C.shape
	Ns = []	 # Nullspaces	

	Ks = []
	Ls = []
	for X, dt in zip(Xs, dts):
		m = X.shape[1]
		t = dt * np.arange(m)

		# Build the constraint matrix
		D_x0s, D_cvec = sensitivity(phis, C, X[:,0], t, **kwargs)

		# Derivative with respect to c
		K = np.vstack(D_cvec)[:,vec_mask]
		# Derivative with respect to initial condition
		L = ss.hstack([np.vstack(D_x0s), 0*ss.eye(d*m, d*(m-1))])
		# statisfy  E^j(x_0) - x_j = 0
		L = L - eye(m * d)

		# Total constraint matrix
		# Note 
		# * the first d columns are zero 
		# * the d leading columns of L are nonzero
		# * L[d:,d:] is the identity matrix
		# A = [ 0  0    0 ]
		#     [ K  L1  -I ]
		A = ss.hstack([K, L])
		
		# Construct the nullspace of A
		# By construction, this spanns
		# N = [ I  0  ]
		#     [ 0  I  ]
		#     [ K  L1 ]
		N = np.hstack([np.vstack(D_cvec[1:])[:,vec_mask], np.vstack(D_x0s[1:])])
		N = np.vstack([np.eye( sum(vec_mask) + d),  N])
		
		N, _ = sl.qr(N, mode = 'economic')
		#err = np.max(np.abs(A @ N))
		#err = np.linalg.norm(A @ N, 'fro')
		#print("err", err)
		#assert err < 1e-10
		return N

def construct_tangent_space_one(C, phis, Xs, dts, mask = None, **kwargs):
	if len(Xs) > 1:
		raise NotImplementedError
	X = Xs[0]
	dt = dts[0]
	t = dt * np.arange(X.shape[1])
	con = Constraint(phis, [t], mask = mask)
	F = con.jac(con.encode(C, Xs)).tolil()	
	n_c = F.shape[1] - F.shape[0]
	K = F[:,:n_c]
	L = F[:,n_c:]
	splu = scipy.sparse.linalg.splu(L.tocsc())
	N = splu.solve(K.A)
	U = np.vstack([np.eye(n_c), -N])
	U, _ = np.linalg.qr(U, mode = 'reduced')
	return U	
