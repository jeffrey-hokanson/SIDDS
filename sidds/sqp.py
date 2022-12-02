import numpy as np
import scipy.optimize
from functools import lru_cache, wraps
#from methodtools import lru_cache
#from functools import wraps
from iterprinter import IterationPrinter

class Termination(Exception):
	pass

class SmallStepTermination(Termination):
	def __init__(self, norm_dx):
		self.norm_dx = float(norm_dx)
	def __repr__(self):
		return f"SmallStepTermiation({self.norm_dx:5e})"

class OptimalTermination(Termination):
	pass

def np_cache(function, maxsize = 16, typed = False):
	r"""
	https://stackoverflow.com/a/52332109/3597894
	"""	

	@lru_cache(maxsize = maxsize, typed = typed)
	def cached_wrapper(*args):
		args = (np.array(arg) for arg in args)
		ret = function(*args)
		if isinstance(ret, np.ndarray):
			ret.setflags(write = False)
		elif isinstance(ret, list):
			for r in ret:
				if isinstance(r, np.ndarray):
					r.setflags(write = False)
		return ret

	@wraps(function)
	def wrapper(*args):
		return cached_wrapper(*(tuple(arg) for arg in args))
		
	wrapper.cache_info = cached_wrapper.cache_info
	wrapper.cache_clear = cached_wrapper.cache_clear
	return wrapper


class NonlinearEqualityConstraint(scipy.optimize.NonlinearConstraint):
	r""" An extension of scipy.optimize.NonlinearConstraint
	"""
	def __init__(self, fun, target, jac, hess = None, maxsize = 16):
		if np.all(target ==  0):
			self._fun = np_cache(fun, maxsize = maxsize)
		else:
			self._fun = np_cache(lambda x: fun(x) - target, maxsize = maxsize)
		
		self._jac = np_cache(jac, maxsize = maxsize)

		if hess is None:
			self._hess = None
		else:
			self._hess = np_cache(hess, maxsize = maxsize)

	def __call__(self, x):
		return self._fun(x)
	
	def fun(self, x):
		return self._fun(x)
	
	def jac(self, x):
		try:
			return self._jac(x)
		except AttributeError:
			raise NotImplementedError

	def hess(self, x, z):
		if self._hess is None:
			raise NotImplementedError
		
		try:
			return self._hess(x, z)
		except:
			raise NotImplementedError 

	@property
	def is_equality_constraint(self):
		return True

class EqualitySQP:
	def solve(self, x0 = None, maxiter = 10):
		if x0 is not None:
			self.x = np.copy(x0)
			self.it = 0

		for it in range(maxiter):
			self.step()
			self.it += 1
	
	def solve_kkt(self, A, B, g, c):
		r""" Solve a KKT system 


		min_p  g.T @ p + 0.5 * p.T @ B @ p
		s.t.   c + A @ p = 0

		corresponding to the linear system

			[B     A.T ] [x] = [-g]
			[A     0   ] [z] = [-c]


		Parameters
		----------
		A: np.ndarray
			Jacobian of the constraints
		B: np.ndarray
			Hessian approximation
		g: np.ndarray
			gradient of the objective
		c: np.ndarray
			Target for the active constraints
			

		Returns
		-------
		dx: np.ndarray
			Search direction in state variables

		z : np.ndarray
			Lagrange multipliers associated with the solution
		"""

		n_con = A.shape[0]
		Z = np.zeros((n_con, n_con))
		AA = np.block([[B, A.T],[A, Z]])
		bb = np.hstack([-g, -c])
		xx = np.linalg.solve(AA, bb)
		return xx[:-n_con], xx[-n_con:] 
		
	
class LiuYuanEqualitySQP(EqualitySQP):
	r"""

	This is Algorithm 2.2

	From "A Sequential Quadratic Programming Method Without a Penalty Function 
		or Filter for Nonlinear Equality Constrained Optimization"
		Xinwei Liu and Yaxiang Yuan
		SISC 2011

	Parameters
	----------
	sigma: float
		Parameter controlling sufficient decrease in the Armijo condition for both objective and constraint
	xi1: float
		Parameter controlling an alternative requirement for sufficient decrease of the objective
	xi2: float
		Parameter controlling an alternative requirement for sufficient decrease of the constraint
	"""
	def __init__(self, fun, jac, hess, constraints, 
			verbose = False, cache = False,
			sigma = 0.01,
			xi1 = 1e-10,
			xi2 = 1e-4,
			kappa1 = 1e5,
			kappa2 = 1e-6,
			tol_dx = 1e-6, tol_h = 1e-6, tol_Ah = 1e-6, tol_opt = 1e-6,
			v_soc = -1,
		):
		if cache:
			self.fun = np_cache(fun)
			self.jac = np_cache(jac)
			self.hess = np_cache(hess)
		else:
			self.fun = fun
			self.jac = jac
			self.hess = hess

		self.constraints = constraints
	
		self.verbose = verbose

		self.sigma = sigma
		self.xi1 = xi1
		self.xi2 = xi2
		self.kappa1 = kappa1
		self.kappa2 = kappa2
		self.tol_dx = tol_dx
		self.tol_h = tol_h
		self.tol_Ah = tol_Ah
		self.tol_opt = tol_opt
		self.v_soc = v_soc

	
		

	def init_constants(self):
		self.vmax = 0
		self.eq3_old = False
		self.r = 0.9
		

	def init_solve(self):
		self.it = 0
		self.init_constants()
	

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
				)

		self._printer.print_header(it = 'iter', obj = 'objective', lagrangian_grad = 'optimality', 
				constraint = 'constraint', alpha = 'step', norm_dx = 'dx', eq1 = 'eq1', eq2 = 'eq2', eq3 = 'eq3',)

		self._step_data = {}
	
	def print_iter(self):
		self._printer.print_iter(**self._step_data)

	def solve(self, x0 = None, z0 = None, maxiter = 10):
		r""" Solve the constrained optimization problem

		
		"""
		self.init_solve()	
		if self.verbose: self.init_printer()
		
		if x0 is not None:
			self.x = np.copy(x0)
			self.it = 0
			self.init_constants()

			if z0 is not None:
				self.z = np.copy(z0)
			else:
				self.z = 0 * self.constraints(self.x)


		for it in range(maxiter):
			try:
				self.step()
				self.it += 1
			except Termination as e:	
				print(repr(e))
				break

	def make_relaxed_constraint(self, x):
		r""" Solve a least squares problem to determine the RHS in the KKT system 
 
		Given a linearization of the constraints

			h(x) \approx h(x_k) + A(x_k) p
		
		Compute an approximate least squares solution to the problem

			min_p \| h(x_k) + A(x_k) p\|_2^2
		
		satifying two additional constraints:
	
		a) \| p_k \| <= kappa_1 \| A(x_k) h\|_2^2

		b) if \| h(x_k)\| \ne 0, 
				\| h(x_k) \| - \| h(x_k) + A(x_k) p_k\| 
					\ge kappa_2 \| A(x_k)^\trans h(x_k)\|/\|h(x_k)\|_2

		These conditions will be checked before the iteration continues.

		Parameters
		----------
		x: np.ndarray
			Location of the current iterate

		Returns
		-------
		dp: np.ndarray
			Approximate least squares solution

		"""
		raise NotImplementedError	
		h = self.constraints(x)
		A = self.constraints.jac(x)
		dp, _, _, _ = np.linalg.lstsq(A, -hx, rcond = None)
		return dp


	def check_relaxed_constraint(self, x, dp):
		r""" Verify constraints satisfy requirements for convergence

		See LY11, p.548, conditions (a) and (b) after eq. (2.2) 
		"""
		h = self.constraints(x)
		A = self.constraints.jac(x)
		
		norm_dp = np.linalg.norm(dp)
		norm_Ah = np.linalg.norm(A.T @ h)
		if norm_dp > self.kappa1 * norm_Ah:
			mess = f"Violated constraint (a) on generating relaxed constraints:\n"
			mess += f" || dp || <= self.kappa || A.T @ h ||\n"
			mess += f"got: {norm_dp:5e} <= {self.kappa1*norm_Ah:5e}\n"
			raise AssertionError(mess)
		
		norm_res = np.linalg.norm(h + A @ dp)
		norm_h = np.linalg.norm(h)

		if norm_h > 1e-5:
			lhs = norm_h - norm_res
			rhs = self.kappa2 * norm_Ah**2 /norm_h
			if lhs < rhs:
				mess ="Violated constraint (b) on generating relaxed constraints:\n"
				mess += " || h || - || h + A dp || >= kappa2 || A.T h||^2/|| h||\n"
				mess += f"got {lhs:5e} >= {rhs:5e} with ||h||={norm_h:5e} "
				raise AssertionError(mess)

	def solve_relaxation(self, x, h, A):
		r""" Compute the right hand side for the relaxed equality constraints

		Approximately solve the linear system

			dp = min_p \| h + A p\ |_2

		This approximate solution must satisfy
		
		a) \| p_k \| <= kappa_1 \| A(x_k) h\|_2^2

		b) if \| h(x_k)\| \ne 0, 
				\| h(x_k) \| - \| h(x_k) + A(x_k) p_k\| 
					\ge kappa_2 \| A(x_k)^\trans h(x_k)\|/\|h(x_k)\|_2



		Parameters
		----------
		h: np.ndarray
			Current values of the equality constraints
		A: np.ndarray or sparse matrix
			Jacobian of the constraints
		
		Returns
		-------
		dp: np.ndarray
			Approximate solution to the linear system
		"""
		dp, _, _, _ = np.linalg.lstsq(A, -h, rcond = None)
		return dp

	def solve_qp(self, x, z, g, h, c, A, x0):
		r""" Solve the quadratic program
		
		Approximately solve the quadratic subproblem

		min_d    g.T @ d + 0.5 * (d.T @ B @ d)
		s.t.     A @ d = c

		This corresponds to solving the KKT system

			[B     A.T ] [x] = [-g]
			[A     0   ] [z] = [c]

		Parameters
		----------

		x0:
			Solution to the least squares step
			Optionally 
		"""
		try:
			B = self.hess(x) + self.constraints.hess(x, z)
		except (NotImplementedError, AttributeError):
			B = self.hess(x)
		n_con = A.shape[0]
		Z = np.zeros((n_con, n_con))
		AA = np.block([[B, A.T],[A, Z]])
		bb = np.hstack([-g, c])
		
		xx = np.linalg.solve(AA, bb)
		return xx[:-n_con], xx[-n_con:] 
		
	def step(self):
		#self._step_data = {}
		x = np.copy(self.x)
		z = np.copy(self.z)

		#############################################################################
		# Compute modification of KKT system
		#############################################################################
		h = self.constraints(x)
		A = self.constraints.jac(x)
		dp = self.solve_relaxation(x, h, A)
		#self.check_relaxed_constraint(x, dp)

		#############################################################################
		# Solve the QP subproblem 
		#############################################################################
		g = self.jac(x)
		
		# constraint A @ dp = A @ p 
		c = A @ dp
		p, z = self.solve_qp(x, z, g, h, c, A, dp)
		p_z = z - self.z

		#############################################################################
		# Check termination conditions
		#############################################################################
		lagrangian_grad = g + A.T @ z
		norm_h = np.linalg.norm(h)
		norm_p = np.linalg.norm(p)
		self._step_data['it'] = self.it
		fx = self._step_data['obj'] = self.fun(x)
		self._step_data['lagrangian_grad'] =  np.linalg.norm(lagrangian_grad)
		self._step_data['constraint'] = norm_h

		if ( 
			(np.linalg.norm(lagrangian_grad) < self.tol_opt) and 
			(norm_h < self.tol_h) and  
			(np.linalg.norm(A.T @ h) > self.tol_Ah * min(norm_h, 1))
			):
			if self.verbose:
				self.print_iter()
			raise OptimalTermination
	
		#############################################################################
		# Line search
		#############################################################################
		alpha = 1
		v = np.linalg.norm(self.constraints(x))
		norm_p = np.linalg.norm(p)
		phi = np.linalg.norm(self.constraints(x) + self.constraints.jac(x) @ p) - v


		it = 0
		soc_step = False
		success = False
		while True:
			if (v <= self.v_soc) and it == 1:
				soc_step = True
			else:
				soc_step = False
			
			if not soc_step and it > 0:
				alpha *= 0.5
			
			xt = x + alpha*p

			if soc_step:
				# correction
				ht = self.constraints(xt)
				dpt = self.solve_relaxation(x, ht, A)
				
				# solve QP for correction step
				# min_{pt}  g.T @ (p + pt) + 0.5 (p + pt).T @ B @ (p + pt)
				# s.t. A @ (p + pt) = A @ dpt 
				pt, zt = self.solve_qp(x, z, g, h, A @ (dpt - p), A, p)
				xt = x + pt		

			ht = self.constraints(xt)
			vt = np.linalg.norm(ht)	


			# LY11, eq. 2.7	
			eq1_lhs = self.fun(xt) - fx 
			eq1_rhs = min(self.sigma*alpha * g.T @ p, -self.xi1*vt)
			eq1 = (eq1_lhs <= eq1_rhs)
			
			
			# LY11, eq. 2.8
			eq2_lhs = vt
			eq2_rhs = max((self.r + 1)/2, 0.95) * self.vmax
			eq2 = (eq2_lhs <= eq2_rhs) or (self.vmax == 0)

			# LY11, eq. 2.9
			eq3_lhs = vt - v
			eq3_rhs = min(self.sigma * alpha * phi, -self.xi2 * alpha**2 * norm_p**2) 
			eq3 = (eq3_lhs <= eq3_rhs)
				
			if soc_step:
				norm_dx = np.linalg.norm(xt - x)
			else:
				norm_dx = alpha * norm_p
	
			self._step_data['alpha'] = alpha
			self._step_data['eq1'] = 'T' if eq1 else 'F'
			self._step_data['eq2'] = 'T' if eq2 else 'F'
			self._step_data['eq3'] = 'T' if eq3 else 'F'
			self._step_data['norm_dx'] = norm_dx	
			
			if ((eq1 and eq2) or eq3) or ( (eq1 and eq2) and soc_step):
				break
	
			if norm_dx < self.tol_dx:
				if self.verbose: self.print_iter()
				raise SmallStepTermination(norm_dx)

			it += 1
		
		self.x = xt
		self.z = self.z + alpha*p_z

		if self.verbose:
			self.print_iter()

		if norm_dx < self.tol_dx:
			raise SmallStepTermination(norm_dx)
	
		
		#############################################################################
		# Update parameters
		#############################################################################
		if eq3 and (not self.eq3_old):
			self.vmax = v
		self.eq3_old = eq3
		if eq3:
			self.r = vt/v
		
		#############################################################################
		# Update tuning parameters
		#############################################################################
		# TODO: add regularization penalty
		# 		+ increase parameter if KKT not solved accurately or step length < 1
		# 		+ decrease parameter if model sufficiently accurate


if __name__ == '__main__':
	# This comes from NW06, 15.4
	def fun(x):
		return 2*(x[0]**2 + x[1]**2 -1) - x[0]
	def grad(x):
		return np.array([4*x[0] - 1, 4*x[1]])
	def hess(x):
		return 4*np.eye(2)

	constraint = NonlinearEqualityConstraint(
		fun = lambda x: np.array([x[0]**2 + x[1]**2 - 1,]),
		target = np.zeros(1),
		jac = lambda x: 2*x.reshape(1,2),
		hess = lambda x, z: z*np.eye(2),
	)	
	x0 = np.ones(2)/np.sqrt(2)
	#x0 = np.array([1.1,0])	
	#opt = LineSearchSQP(fun, jac, hess, constraint)
	if False:
		res = scipy.optimize.minimize(fun, x0, jac = grad, hess =hess, constraints = constraint, method = 'trust-constr')
		print(res)
	if False:
		opt = LineSearchEqualitySQP(fun, grad, hess, constraint)
		opt.penalty = 1e1
		opt.solve(x0, maxiter = 100)	 
	
	opt = LiuYuanEqualitySQP(fun, grad, hess, constraint)
	dp = opt.make_relaxed_constraint(x0)
	opt.solve(x0, maxiter = 100)




