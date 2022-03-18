import numpy as np
from .sensitivity import sensitivity as _sensitivity
from .sqp import *
from copy import deepcopy as copy
from .lide import encode, decode
from .matrices import vec, unvec
from .forward import evolve_system
import scipy.sparse

def evolve(phis, C, x0, dt):
	return evolve_system(phis, C, x0, [0, dt])[:,-1]

def sensitivity(phis, C, x0, dt):
	Ws, Vs =  _sensitivity(phis, C, x0, [0, dt])
	return Vs[-1], Ws[-1]

class Constraint(NonlinearEqualityConstraint):
	def __init__(self, phis, ts, mask = None):
		self.phis = copy(phis)
		self.ts = copy(ts)
	
		n = len(self.phis)
		d = self.phis[0].dim
		if mask is None:
			mask = np.ones((d, n), dtype = np.bool)
		self.mask = mask
	
	@property
	def dim(self):	
		r"""The number of evolving variables
		"""
		return self.phis[0].dim

	@property
	def data_dim(self):
		return [ (self.dim, len(t)) for t in self.ts]
	
	def encode(self, C, Ys):
		return encode(C, Ys, self.mask)

	def decode(self, x):
		return decode(x, self.data_dim, self.mask)


	def fun(self, x):
		C, Zs = self.decode(x)
		vec_c = vec(C)
		h = []
		for Z, t in zip( Zs, self.ts):
			for j in range(Z.shape[1]-1):
				zpred = evolve(self.phis, C, Z[:,j], t[j+1]-t[j])
				h += [zpred - Z[:,j+1]]
		return np.hstack(h)	
		
	def jac(self, x):
		C, Zs = self.decode(x)
		d, n = C.shape
		vec_mask = vec(self.mask)
		if len(Zs) > 1: raise NotImplementedError
		Vs = []
		Ws = []
		for Z, t in zip( Zs, self.ts):
			for j in range(Z.shape[1]-1):
				V, W = sensitivity(self.phis, C, Z[:,j], t[j+1] - t[j])
				Vs += [ V[:,vec_mask]  ]
				Ws += [ W ]
		#print(len(Vs))	
		V = np.vstack(Vs)
		W_diag = scipy.sparse.block_diag(Ws)
		zero = scipy.sparse.csr_matrix((V.shape[0], d))
		A = scipy.sparse.hstack([V, W_diag, zero])
		I = scipy.sparse.diags(np.ones(A.shape[0]), offsets = V.shape[1] + d,shape = A.shape)
		A -= I
		return A
