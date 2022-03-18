import numpy as np
import abc

r"""

Data formats


X, Y 
	    [   .        .         ]
	X = [ x(t_0)  x(t_1)  ...  ]
		[   .        .         ]

"""


################################################################################
# Basis functions
################################################################################

class BasisFunction(abc.ABC):
	r""" An abstract base class for use with 
	"""
	@abc.abstractmethod
	def eval(self, x):
		r"""

		Parameters
		----------
		x: array-like
			Array of shape (# points, dim)
		"""
		pass

	@abc.abstractmethod
	def grad(self, x):
		pass

	@property
	@abc.abstractmethod
	def dim(self):
		pass

	def _format_input(self, x):
		x = np.array(x)
		assert len(x.shape) == 2, "Must provide a two-dimensional array"
		assert x.shape[0] == self.dim, "First dimension must be the ambient dimension of the space"
		return x

class Monomial(BasisFunction):
	r""" Construct a polynomial with only one term

	Given a degree (a list of non-negative integers), 
	constructs the function

		p(x) = \prod_j x_j^{d_j}

	"""
	def __init__(self, degree):
		self.degree = np.array(degree, dtype = int).flatten()
	
	@property
	def dim(self):
		return self.degree.shape[0]

	def eval(self, x):
		x = self._format_input(x)
		fx = np.ones(x.shape[1], dtype = x.dtype)

		for k, p in enumerate(self.degree):
			if p > 0:
				fx *= x[k,:]**p

		return fx
	
	def grad(self, x):
		x = self._format_input(x)
		gx = np.ones((x.shape[1], self.dim), dtype = x.dtype)
	
		for j in range(self.dim):
			for k, p in enumerate(self.degree):
				if k == j and p == 0:
					gx[:,j] = 0
				elif k == j:
					gx[:,j] *= p*(x[j,:]**(p-1))
				else:
					gx[:,j] *= x[k,:]**p
						
		return gx

#	def hess(self, x):
#		x = self._format_input(x)

