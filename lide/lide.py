import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.linalg import solve_triangular, LinAlgError
import time
from lide.matrices import *
from .lsopinf import lsopinf
from .sqp import * 
from copy import deepcopy as copy
from .pareto import *
from .symmlq import Symmlq
from .denoise import *
from functools import partial
from scipy.interpolate import interp1d
		

class Counter:
	r""" Simple counter class to count KKT iterations
	"""
	def __init__(self, *args, **kwargs):
		self.count = 0
	def __call__(self, *args, **kwargs):
		self.count += 1

################################################################################
# Preconditioners
################################################################################

class CholeskyInverse(scipy.sparse.linalg.LinearOperator):
	r""" Constructs an inverse operator for a symmetric positive definite matrix

	This class constructs an inverse operator using a Cholesky decomposition.
	Note, if this is called sufficiently many times, it may be more efficient
	to compute the inverse explicitly.

	Parameters
	----------
	A: np.ndarray
		An (n,n) dimensional array, symmetric and positive definite
	"""
	def __init__(self, A):
		self.L = scipy.linalg.cholesky(A, lower = True)
	
	@property
	def shape(self):
		return self.L.shape

	def _matmat(self, x):
		r""" Returns A^{-1} x
		"""
		y = solve_triangular(self.L, x, lower = True)
		z = solve_triangular(self.L, y, lower = True, trans = 'T')
		return z


class ExplicitInverse(scipy.sparse.linalg.LinearOperator):
	r""" Apply the explicit inverse 

	Although inverting matrices is generally to be avoided,
	if the matrix-vector product is going to be called many times,
	it may be more efficient to explicitly form the matrix inverse
	rather than calling back substitution multiple times.

	"""
	def __init__(self, A):
		self.inv = scipy.linalg.inv(A)

	@property
	def shape(self):
		return self.inv.shape

	def _matmat(self, x):
		return self.inv @ x


class SparseLUInverse(scipy.sparse.linalg.LinearOperator):
	def __init__(self, A, **kwargs):
		self.lu = scipy.sparse.linalg.splu(A.tocsc(), **kwargs)
		self._shape = A.shape

	@property
	def shape(self):
		return self._shape
	
	def _matmat(self, x):
		return self.lu.solve(x)

class BlockDiagonalMatrix(scipy.sparse.linalg.LinearOperator):
	r""" Construct a block diagonal matrix from operators
	"""
	def __init__(self, *blocks):
		self.blocks = blocks
		self.dim = np.cumsum([0]+[block.shape[0] for block in self.blocks])
	
	@property	
	def shape(self):
		return (self.dim[-1], self.dim[-1])

	def _matvec(self, x):
		return np.hstack([B @ x[d1:d2] for B, d1, d2 in zip(self.blocks, self.dim[0:-1], self.dim[1:])])
	def _matmat(self, x):
		return np.vstack([B @ x[d1:d2] for B, d1, d2 in zip(self.blocks, self.dim[0:-1], self.dim[1:])])
	


################################################################################
# Utility Functions
################################################################################

def encode(C, Ys, mask = None):
	if mask is None:
		return np.hstack([vec(C)] + [vec(Y) for Y in Ys])
	else:
		return np.hstack([vec(C)[vec(mask)]] + [vec(Y) for Y in Ys])

def decode(x, dims, mask = None):
	total = sum([dim[0]*dim[1] for dim in dims])
	d = dims[0][0]

	if mask is None:
		C_shape = ( (len(x) - total)//d, d)
		C = unvec(x[0:C_shape[0]*C_shape[1]], d)
		start = C_shape[0]*C_shape[1]
	else:
		flat_mask = vec(mask)
		flat_C = np.zeros(mask.shape[0]*mask.shape[1])
		start = np.count_nonzero(mask) 
		flat_C[flat_mask] = x[:start]
		C = unvec(flat_C, d)

	Xs = []
	for dim in dims:
		Xs += [unvec(x[start:start+dim[0]*dim[1]], d)]
		start += dim[0]*dim[1]
	return C, Xs

def mask_matrix(mask):
	mask_flat = vec(mask)
	row = np.argwhere(mask_flat).flatten()
	col = np.arange(len(row))
	M = scipy.sparse.coo_matrix((np.ones(col.shape), (row, col)), shape = (len(mask_flat), len(col)))
	return M.tocsr()


################################################################################
# Specify the equality constraints
################################################################################

class LIDENonlinearEqualityConstraint(NonlinearEqualityConstraint):
	def __init__(self, phis, Xs, dts, points, mask = None, make_DM = make_DM_square):
		self.phis = copy(phis)
		self.dts = np.copy(dts)
		self.points = points
		self.dims = [X.shape for X in Xs]
		
		if mask is None:
			mask = np.ones((self.dim, self.N_phis), dtype = np.bool)
		self.mask = mask
		
		self.DMs = [make_DM(X.shape[1], self.points) for X in Xs]

	def encode(self, C, Ys):
		return encode(C, Ys, self.mask)

	def decode(self, x):
		return decode(x, self.dims, self.mask)

	def __call__(self, x):
		return self.fun(x)
	
	@property
	def dim(self):
		return self.phis[0].dim

	@property
	def N_phis(self):
		return len(self.phis)

	def fun(self, x):
		C, Xs = self.decode(x)
		hs = []
		for X, dt, DM in zip(Xs, self.dts, self.DMs):
			#D, M = make_DM(X.shape[1], self.points)	
			D, M = DM
			Phi = make_Phi(X, self.phis)
			h = (D/dt) @ X.T - M @ (Phi @ C.T)
			hs += [vec(h.T)]
		
		return np.hstack(hs)

	def jac(self, x):
		C, Xs = self.decode(x)
		dim = C.shape[0]
		Mask = mask_matrix(self.mask)
		A = []
		for i, (X, dt, DM) in enumerate(zip(Xs, self.dts, self.DMs)):
			D, M = DM
			vec_D = make_vec(D, dim)
			vec_M = make_vec(M, dim)	
			
			Phi = make_Phi(X, self.phis)
			vec_Phi = make_vec(Phi, dim)
			vec_grad_Phi_c = make_grad_vec_Phi(X, self.phis, C)
			
			A += [ [-vec_M @ vec_Phi @ Mask] + [None, ]*i + [vec_D/dt - vec_M @ vec_grad_Phi_c] + [None, ]*(len(Xs) - i)]
		return scipy.sparse.bmat(A)
				

################################################################################
# Specify the objective 
################################################################################

class LIDEObjective:
	def __init__(self, phis, Ys, Sigmas, dts, points, alpha = 0, epsilon = 1, q = 2,  mask = None, irls = False):
		self.phis = copy(phis)
		self.dts = np.copy(dts)
		self.points = points
		self.Ys = [np.copy(Y) for Y in Ys]	
		self.Sigmas = Sigmas
		self.alpha = alpha
		self.epsilon = max(epsilon, 0)
		self.q = q
		self.irls = irls
		
		if mask is None:
			mask = np.ones((self.dim, self.N_phis), dtype = np.bool)

		self.mask = mask
		self.C_old = np.zeros((self.dim, self.N_phis))
	
	@property	
	def dims(self):
		return [Y.shape for Y in self.Ys]
	
	def encode(self, C, Ys):
		return encode(C, Ys, self.mask)

	def decode(self, x):
		return decode(x, self.dims, self.mask)
	
	@property
	def dim(self):
		return self.phis[0].dim

	@property
	def N_phis(self):
		return len(self.phis)

	def objective(self, x):
		C, Xs = self.decode(x)
		obj = 0
		for Sigma, X, Y, dt in zip(self.Sigmas, Xs, self.Ys, self.dts):
			vec_delta = vec(X - Y)
			obj += 0.5 * (vec_delta.T @ Sigma @ vec_delta)
		return obj

	def regularization(self, x):
		C, Xs = self.decode(x)
		Mask = mask_matrix(self.mask)
		vec_C = vec(C) @ Mask
		return np.sum(vec_C**2 * (vec_C**2 + self.epsilon)**(self.q/2. - 1))

	def fun(self, x):
		C, Xs = self.decode(x)
		dim = C.shape[0]
		Mask = mask_matrix(self.mask)
		vec_C = vec(C) @ Mask
		vec_C_old = vec(self.C_old) @ Mask
	
		if self.alpha > 0:	
			if self.irls:
				obj = self.alpha * np.sum( vec_C**2 * np.power(vec_C_old**2 + self.epsilon, self.q/2. - 1))
			else:
				obj = self.alpha * np.sum( vec_C**2 * np.power(vec_C**2 + self.epsilon, self.q/2. - 1))		
		else:
			obj = 0


		for Sigma, X, Y, dt in zip(self.Sigmas, Xs, self.Ys, self.dts):
			vec_delta = vec(X - Y)
			obj += 0.5 * (vec_delta.T @ Sigma @ vec_delta)
	
		return obj

	def jac(self, x): 
		C, Xs = self.decode(x)
		Mask = mask_matrix(self.mask)
		vec_C = vec(C) @ Mask
		vec_C_old = vec(self.C_old) @ Mask

		if self.alpha > 0:	
			if self.irls:	
				gs = [2*self.alpha * vec_C * np.power(vec_C_old**2 + self.epsilon, self.q/2. - 1) ]
			else:
				gs = [self.alpha * vec_C * np.power(vec_C**2 + self.epsilon, self.q/2. - 1) * (2*self.epsilon + self.q*vec_C**2)]
		else:
			gs = [np.zeros_like(vec_C)]

		for Sigma, X, Y, dt in zip(self.Sigmas, Xs, self.Ys, self.dts):
			gs += [Sigma @ vec(X - Y)]
				
		return np.hstack(gs)
			
	
	def hess(self, x):
		C, Xs = self.decode(x)
		dim = C.shape[0]
		Mask = mask_matrix(self.mask)
		vec_C = vec(C) @ Mask
		vec_C_old = vec(self.C_old) @ Mask

		# second derivative of penalty
		
		if self.alpha > 0:
			if self.irls:
				h = 2*self.alpha * np.power(vec_C_old**2 + self.epsilon, self.q/2. - 1) 
			else:
				h = self.alpha*(self.epsilon + vec_C**2)**(self.q/2 - 3) * (
					2*self.epsilon**2 + self.epsilon * (5 * self.q - 6) * vec_C**2 + (self.q - 1)*self.q * vec_C**4)
		else:
			h = np.zeros_like(vec_C)

		H = scipy.sparse.block_diag([scipy.sparse.diags(h)] + self.Sigmas)

		return H



class LIDE(LiuYuanEqualitySQP):
	r"""

	Parameters
	----------
	stab_coef: float
		Coefficient multiplying the stabilization parameter in the KKT system.
		Set to zero to disable stabilization
	kkt_reg: float
		add this constant times the identity to the (1,1) block in the KKT system
	relax_reg: float
		regularization added to solve for the p_x component in the relaxation step	
	"""
	def __init__(self, phis, Ys, dts, points = 9, Sigmas = None, mu = 1e2, mask = None,
		solver = 'minres', irls = True, square = True, kkt_reg = 1e-4, relax_reg = 1e-6, solve_qp_kwargs = {}, 
		stab_coef = 1e-4, oversample = 1, **kwargs):
	
		self.kwargs = kwargs
		self.points = points
		self.kkt_reg = kkt_reg
		assert self.kkt_reg >= 0
		self.mu = mu
		assert self.mu > 0


		self.stab_coef = float(stab_coef)
		assert self.stab_coef >= 0

		self.relax_reg = relax_reg
		assert self.relax_reg >=0

		self.oversample = int(oversample)
		assert self.oversample > 0

		if Sigmas is None:
			Sigmas = [scipy.sparse.eye(Y.shape[0]*Y.shape[1]) for Y in Ys]


		self.Ys_orig = Ys
		self.dts_orig = dts

		if self.oversample > 1:
			Ys_new = []
			dts_new = []
			Sigmas_new = []
			for dt, Y, Sigma in zip(dts, Ys, Sigmas):
				dim, M = Y.shape
				t = dt*np.arange(M)
				fit = interp1d(t, Y, kind = 'nearest')
				M_new = M * self.oversample - (self.oversample - 1)
				t_new = (dt/self.oversample)*np.arange( M_new)
				idx_new = np.arange(0, M_new, oversample)
				Ys_new += [fit(t_new)]
				dts_new += [dt/self.oversample]
				Sigma_ = Sigma.tocoo()
				row, col, val = Sigma_.row, Sigma_.col, Sigma_.data
				#  idx // dim <- time index
				#  idx %  dim <- coordinate
				row = (row//dim)*(dim * oversample) + (row % dim) 
				col = (col//dim)*(dim * oversample) + (col % dim) 
				Sigma_new = scipy.sparse.coo_matrix((val, (row, col)), shape = (dim*M_new, dim*M_new))
				Sigmas_new += [Sigma_new]
			
			Ys = Ys_new
			dts = dts_new
			Sigmas = Sigmas_new

		self.Sigmas = Sigmas
		self.Ys = Ys
		self.phis = phis
		self.dts = dts

		self._obj = LIDEObjective(phis, Ys, self.Sigmas, dts, points, mask = mask, irls = irls)
		if square:
			self._constraints = LIDENonlinearEqualityConstraint(phis, Ys, dts, points, mask = mask, make_DM = make_DM_square)
		else:
			self._constraints = LIDENonlinearEqualityConstraint(phis, Ys, dts, points, mask = mask, make_DM = make_DM)
			
		if mask is None:
			mask = np.ones((self.dim, self.N_phis), dtype = np.bool)
		self.mask = mask


		if solver == 'dense':
			self.solve_qp = self.solve_qp_dense
		elif solver == 'minres':
			self.solve_qp = partial(self.solve_qp_minres, **solve_qp_kwargs)
		elif solver == 'symmlq':
			self.solve_qp = partial(self.solve_qp_symmlq, **solve_qp_kwargs)
		elif solver == 'gmres':
			self.solve_qp = partial(self.solve_qp_gmres, **solve_qp_kwargs)
				

		LiuYuanEqualitySQP.__init__(self, self._obj.fun, self._obj.jac, self._obj.hess, self._constraints, **kwargs)	
	
		# Used for printing, but needs to be initialized 
		self._step_data = {}


	def objective(self, Xs = None):
		if Xs is None:
			Xs = self.Xs
		x = self.encode(self.C, Xs)
		return self._obj.objective(x)

	def regularization(self, C = None):
		if C is None:
			C = self.C
		x = self.encode(C, self.Xs)
		return self._obj.regularization(x)
		
	def encode(self, *args):
		return self._obj.encode(*args)
		
	def decode(self, *args):
		return self._obj.decode(*args)


	@property
	def dim(self):
		return self.phis[0].dim

	@property
	def dims(self):
		return [Y.shape for Y in self.Ys]

	@property
	def N_phis(self):
		return len(self.phis)


	@property
	def W(self):
		return self._obj.W

	@W.setter
	def W(self, W):
		self._obj.W = W


	@property
	def alpha(self):
		return self._obj.alpha

	@alpha.setter
	def alpha(self, alpha):
		self._obj.alpha = alpha

	@property
	def epsilon(self):
		return self._obj.epsilon

	@epsilon.setter
	def epsilon(self, epsilon):
		self._obj.epsilon = epsilon

	@property
	def q(self):
		return self._obj.q
	
	@q.setter
	def q(self, q):
		self._obj.q = q
	
	
	@property
	def n_param(self):
		return int(np.sum(self.mask))

	@property
	def mask(self):
		return self._obj.mask
	
	@mask.setter
	def mask(self, mask):
		self._obj.mask = mask
		self._constraints.mask = mask


	def init_printer(self):
		if not self.verbose: return
		self._printer = IterationPrinter(
				it = '4d', 
				obj = '20.10e',
				constraint = '8.2e',
				lagrangian_grad = '8.2e',
				alpha = '8.2e',
				norm_dx = '8.2e',
				eq1 = '5s',
				eq2 = '5s',
				eq3 = '5s',
				kkt_res = '8.2e',
				kkt_iter = '6d',
				epsilon = '8.2e',
				)

		self._printer.print_header(it = 'iter', obj = 'objective', lagrangian_grad = 'optimality', 
				constraint = 'constraint', alpha = 'step', norm_dx = 'dx', eq1 = 'eq1', eq2 = 'eq2', eq3 = 'eq3',
				kkt_res = 'KKT Res.', kkt_iter = 'KKT iter', epsilon = 'epsilon')

		self._step_data = {}




	def solve_relaxation(self, x, h, A):

		At = A.tocsc()	
		n = self.n_param
		K = At[:,:n]
		L = At[:,n:]

		x_c = x[:n]
		
		# [ K ] @ (p_c)        = -h
		# [ d ] @ (x_c + p_c)  = 0

		# Solve: min_{p_c} || h + K @ p_c ||
		p_c, res, rank, s = scipy.linalg.lstsq(K.A, -h)
			

		# Solve: min_{p_x} || h + K @ p_c + L @ p_x ||
		ht = h + K @ p_c

		# We have a square system and no regularization
		if L.shape[0] == L.shape[1] and self.relax_reg == 0:
			try:
				lu = scipy.sparse.linalg.splu(L.tocsc())
				p_x = lu.solve(-ht)
				return np.hstack([p_c, p_x])
			except RuntimeError:
				# if we can't perform LU successfully, default to normal equations
				pass		
		
		# Solve via normal equations w/ regularization
		if L.shape[0] >= L.shape[1]:
			# Over determined constraints
			AA = (self.relax_reg*scipy.sparse.eye(L.shape[0]) + L.T @ L).tocsc()
			lu = scipy.sparse.linalg.splu(AA)
			p_x = lu.solve(-L.T @ ht)
		else:
			# Underdetermined constraints
			AA = (self.relax_reg*scipy.sparse.eye(L.shape[0]) + L @ L.T).tocsc()
			lu = scipy.sparse.linalg.splu(AA)
			p_x = L.T @ lu.solve(-ht)
		
		return np.hstack([p_c, p_x])

	
			
	def build_kkt_system(self, x, z, g, h, c, A, B):
		bb = np.hstack([ -g, c])
		# Build the KKT system
		Bhat = B + self.kkt_reg * scipy.sparse.eye(B.shape[0])
		
		if self.stab_coef > 0:
			# Set the coefficient for the (2,2) block of the KKT system
			# This follows Wri05, eq 4.1 & 4.2 (An Algorithm for Degenerate Nonlinear Programming with Rapid Local Convergence, Stephen Wright)
			# See also GR13, top p. 1992; Wri02, eq. 6.7
			lagrange_grad = - g - A.T @ z
			mu = np.linalg.norm(np.hstack([lagrange_grad, h]), 1)
			mu = self.stab_coef * min(mu, 1)
			C = - mu * scipy.sparse.eye(A.shape[0])
		else:
			C = None
		AA = scipy.sparse.bmat([[Bhat,  A.T], [A, C]]).tocsc()
		return AA, bb

	def build_kkt_preconditioner(self, x, z, g, h, c, A, B):
		B = (B + self.kkt_reg * scipy.sparse.eye(B.shape[0])).tocsr()
		Mask = mask_matrix(self.mask)
		n = Mask.shape[1]
		At = A.tocsc()
		K = At[:,:n]
		L = At[:,n:]

		M1 = ExplicitInverse((B[:n,:n] + (self.mu*K.T @ K).A))
		M2 = SparseLUInverse(B[n:,n:] + self.mu * L.T @ L, permc_spec = 'NATURAL')	
		M3 = self.mu * scipy.sparse.eye(L.shape[0])
		
		MM = BlockDiagonalMatrix(M1, M2, M3)
		return MM

	def solve_qp_dense(self, x, z, g, h, c, A, x0, **kwargs):
		n = len(g)
		B = self.hess(x)
		AA, bb = self.build_kkt_system(x, z, g, h, c, A, B)
		qq = scipy.linalg.solve(AA.A, bb, overwrite_a = True, overwrite_b = True, assume_a = 'sym')
		self._step_data['kkt_res'] = np.linalg.norm(AA @ qq - bb)/np.linalg.norm(bb)	
		return qq[:n], qq[n:]

	def solve_qp_minres(self, x, z, g, h, c, A, x0, **kwargs):
		B = self.hess(x) 
		AA, bb = self.build_kkt_system(x, z, g, h, c, A, B)
		n = len(g)

		# Setup options
		if 'tol' not in kwargs:
			kwargs['tol'] = 1e-10
		if 'maxiter' not in kwargs:
			kwargs['maxiter'] = 500 
		if 'verbose' in kwargs and kwargs['verbose'] == True:
			del kwargs['verbose']
			class MyCounter(Counter):
				def __init__(self, A, b):
					self.A = A
					self.b = b
					Counter.__init__(self)

				def __call__(self, xk):
					Counter.__call__(self)
					res = np.linalg.norm( self.A @ xk - self.b)
					print(f'it : {self.count:5d} \t res {res:5e}')
		else:
			MyCounter = Counter

		MM = self.build_kkt_preconditioner(x, z, g, h, c, A, B)
		q0 = np.hstack([x0, z])
	

		kwargs['callback'] = MyCounter(AA, bb)
		qq, info = scipy.sparse.linalg.minres(AA, bb, x0 = q0, M = MM, **kwargs)
		
		self._step_data['kkt_res'] = np.linalg.norm(AA @ qq - bb)/np.linalg.norm(bb)	
		self._step_data['kkt_iter'] =kwargs['callback'].count
		
		return qq[:n], qq[n:]
	
	def solve_qp_symmlq(self, x, z, g, h, c, A, x0, **kwargs):
		B = self.hess(x) 
		AA, bb = self.build_kkt_system(x, z, g, h, c, A, B)
		n = len(g)
		
		# Setup options
		if 'rtol' not in kwargs:
			kwargs['rtol'] = 1e-14
			
		MM = self.build_kkt_preconditioner(x, z, g, h, c, A, B)
		q0 = np.hstack([x0, z])
		rr = bb - AA @ q0	
		sol = Symmlq(AA, precon = MM, rtol = 1e-14)
		sol.solve(rr, check = False, **kwargs)
		qq = sol.bestSolution + q0
		self._step_data['kkt_res'] = np.linalg.norm(AA @ qq - bb)/np.linalg.norm(bb)	
		self._step_data['kkt_iter'] = sol.nMatvec
		return qq[:n], qq[n:]
		
	def solve_qp_gmres(self, x, z, g, h, c, A, x0, **kwargs):
		B = self.hess(x) 
		AA, bb = self.build_kkt_system(x, z, g, h, c, A, B)
		n = len(g)
		
		# Setup options
		if 'tol' not in kwargs:
			kwargs['tol'] = 1e-10
		if 'maxiter' not in kwargs:	
			kwargs['maxiter'] = 500
		if 'restart' not in kwargs:
			kwargs['restart'] = 1
		
		kwargs['callback'] = Counter()
		MM = self.build_kkt_preconditioner(x, z, g, h, c, A, B)
		q0 = np.hstack([x0, z])
		
		qq, info = scipy.sparse.linalg.gmres(AA, bb, x0 = q0, M = MM, **kwargs)
		self._step_data['kkt_res'] = np.linalg.norm(AA @ qq - bb)/np.linalg.norm(bb)	
		self._step_data['kkt_iter'] =kwargs['callback'].count
	
		return qq[:n], qq[n:]


	def init_solve(self, Xs0 = None, C0 = None, smooth = None, ):
		LiuYuanEqualitySQP.init_solve(self)
		
		# Generate initial conditions
		if Xs0 is None:
			try:
				Xs0 = self.Xs
			except AttributeError:
				if smooth is None:
					Xs0 = [np.copy(Y) for Y in self.Ys]
				else:
					Xs0 = []
					for X, dt in zip(self.Ys_orig, self.dts_orig):
						# Denoise the original data
						X0 = denoise_fixed(X, dt, lam = smooth)	
						if self.oversample > 1:
							dim, M = X.shape
							t = dt*np.arange(M)
							# If oversampling, interpolate on smoothed data
							fit = interp1d(t, X0, kind = 'cubic')
							M_new = M * self.oversample - (self.oversample - 1)
							t_new = (dt/self.oversample)*np.arange( M_new)
							X0 = fit(t_new)
						Xs0 += [X0]
						
	
		if C0 is None:
			try:
				C0 = self.C
			except AttributeError:
				# TODO: Better initializaiton calls
				#C0 = np.zeros((self.dim, self.N_phis))
				C0 = lsopinf(self.phis, Xs0, self.dts, self.points, verbose = False)	

		return C0, Xs0	


	def solve(self, Xs0 = None, C0 = None, maxiter = 100, mask = None, smooth = None, history = False):
		C0, Xs0 = self.init_solve(C0 = C0, Xs0 = Xs0, smooth = smooth)
		self.init_printer()
		
		self.alpha = 0
		self.epsilon = None
		
		if mask is not None:
			self.mask = np.copy(mask)

		x0 = self.encode(C0, Xs0)
		self.x = np.copy(x0)
		self.z = self.constraints.fun(x0)
		
		if history:
			history = []
		else:
			history = None
		for it in range(maxiter):
			try:	
				self._step_data['epsilon'] = self.epsilon
				self.step()
			except Termination as e:
				if self.verbose:
					print(repr(e))
				break
			
			if history is not None:
				history += [{
					'C': np.copy(self.C), 
					'Xs': np.copy(self.Xs), 
					'it': self.it, 
					'lagrangian_grad': np.copy(self._step_data['lagrangian_grad']),
					'constraint': np.copy(self._step_data['constraint']),
				}]
			
			self.it += 1

		return history

	def solve_threshold(self, Xs0 = None, C0 = None, maxiter = 100, threshold = 1e-2, smooth = None):
		C0, Xs0 = self.init_solve(Xs0 = Xs0, C0 = C0, smooth = smooth)
		self.init_printer()
		x0 = self.encode(C0, Xs0)
		self.x = np.copy(x0)
		self.z = self.constraints.fun(x0)

		self.it = 0
		for it in range(maxiter):
			try:
				self.step()	
			except Termination as e:
				mask = np.abs(self.C) >= threshold
				if np.array_equal(mask, self.mask):
					if self.verbose: print(repr(e))
					break
			
				self.mask = mask	
				C, Xs = self.decode(self.x)
				Xs_old = np.copy(Xs)
				self._obj.mask = self.mask
				self.constraints.mask = self.mask
				self.x = self.encode(C, Xs)
				#C0, Xs0 = self.init_solve()
				Xs_new = self.Xs

			self.it += 1
		
	def solve_irls(self, C0 = None, Xs0 = None, maxiter = 40, 
		q = 0, alpha = 1, epsilon = 1, epsilon_step = 0.1, epsilon_min = 1e-8, smooth = None,
		history = False):
		r"""
		Parameters
		----------
		maxiter: int
			maximum number of iterations for each epsilon level
		q: float 
			which q-norm is used as a penalty
		alpha: float
			weight attached to the IRLS penalty term
		epsilon: float
			starting epsilon
		epsilon_step: float
			How much epsilon is decreased on termination
			
		"""
		C0, Xs0 = self.init_solve(Xs0 = Xs0, C0 = C0, smooth = smooth)
		self.init_printer()

	
		# Turn off augmentation for the objective
		self._obj.mu = 0	
		self.x = np.copy(self.encode(C0, Xs0))
		self.z = self.constraints.fun(self.x)
		self.alpha = alpha
		self.q = q

		if epsilon is not None:
			self.epsilon = epsilon

		self.it = 0
		self.it_epsilon = 0	
		if history:
			history = []
		else:
			history = None

		while True:
			self._obj.C_old = np.copy(self.C)
			self._step_data['epsilon'] = self.epsilon	
			
			try:
				self.step()
				update_epsilon = False
			except Termination as e:
				update_epsilon = True
			
			if history is not None:
				history += [{
					'C': np.copy(self.C), 
					'Xs': np.copy(self.Xs), 
					'it': self.it, 
					'epsilon':np.copy(self.epsilon),
					'lagrangian_grad': np.copy(self._step_data['lagrangian_grad']),
					'constraint': np.copy(self._step_data['constraint']),
				}]

			if self.it_epsilon >= maxiter:
				update_epsilon = True

			if update_epsilon:
				self.epsilon = epsilon_step*self.epsilon
				self.init_constants()	# reset the constants associated with this problem
				self.it_epsilon = 0
				if self.epsilon <= epsilon_min:
					break
		
			self.it += 1
			self.it_epsilon += 1
		
		return history

	def solve_polish(self, **kwargs):
		nnz = int(np.round(self.regularization()))
		if self.verbose:
			print(f"Identified {nnz:d} non-zero coefficients")
		threshold = np.sort(np.abs(self.C.flatten()))[-nnz]
		mask = np.abs(self.C) >= threshold
		return self.solve(C0 = self.C * mask, mask = mask, **kwargs)
			
	
	@property
	def C(self):
		return self.decode(self.x)[0]

	@property
	def Xs(self):
		return self.decode(self.x)[1]



def lide_L_corner(phis, Ys, dts, alpha_min = 1e-5, alpha_max = 1e2, q = 0, maxiter = 40, epsilon0 = 1e0, tol = 1e-2, **kwargs):
	r"""
	"""
	def solve_lide(alpha, lide = None, **bonus):
		if lide is not None:
			C0 = lide.C
			Xs0 = lide.Xs
		else:
			C0 = None
			Xs0 = None
		lide = LIDE(phis, Ys, dts, **kwargs)
		lide.solve_irls(C0 = C0, Xs0 = Xs0, maxiter = maxiter, q = q, alpha = alpha, epsilon = epsilon0, epsilon_min = 1e-10)

		obj = lide.objective(lide.Xs)
		reg = lide.regularization(lide.C)

		return {'obj': obj, 'reg': reg, 'kwargs': {'lide': lide}}
		

	alpha1, alpha2, ret2, ret3 = pareto_corner(solve_lide, alpha_min, alpha_max, tol = tol) 

	return ret2['kwargs']['lide']




def lide_covariance_C(C, phis, Xs, dts, mask = None, points = 9, Sigmas = None):
	constraint = LIDENonlinearEqualityConstraint(phis, Xs, dts, points, mask = mask)
	F = constraint.jac(constraint.encode(C, Xs))
	U = scipy.linalg.null_space(F.A)
	if Sigmas is None:
		Sigmas = [scipy.sparse.eye(X.shape[0]*X.shape[1]) for X in Xs]
	
	obj = LIDEObjective(phis, Xs, Sigmas, dts, points, alpha = 0, mask = mask)
	J = obj.hess(obj.encode(C, Xs))
	UJU = U.T @ J @ U 
	V, s, VT = np.linalg.svd(UJU, full_matrices = False)
	S_C = np.eye(U.shape[0], np.sum(obj.mask))
	#inv_UJU = V @ np.diag(1./s) @ V.T
	VUS = V.T @ (U.T @ S_C)
	cov_C = VUS.T @ (np.diag(1./s) @ VUS) 
	return cov_C

def lide_bias_C(C, phis, Xs, dts, mask = None, points = 9, Sigmas = None):
	constraint = LIDENonlinearEqualityConstraint(phis, Xs, dts, points, mask = mask)
	obj = LIDEObjective(phis, Xs, Sigmas, dts, points, alpha = 0, mask = mask)
	n_c = np.sum(obj.mask)
	F = constraint.jac(constraint.encode(C, Xs)).A
	h = constraint.fun(constraint.encode(C, Xs))
	print("constraint violation", np.linalg.norm(h))
	cz = np.linalg.lstsq(F, -h, rcond = None)[0]
	return cz[:n_c]


