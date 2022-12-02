# An interface into pysindy
import numpy as np
from copy import deepcopy as copy
import pysindy as ps
from pysindy.feature_library.base import BaseFeatureLibrary


class Library(BaseFeatureLibrary):
	def __init__(self, phis):
		self.phis = phis

	def fit(self, X, y = None):
		return self

	@property
	def n_features_in_(self):
		return self.phis[0].dim

	@property
	def n_output_features_(self):
		return len(self.phis)

	def transform(self, X):
		return np.vstack([phi.eval(X.T) for phi in self.phis]).T

	def get_feature_names(self, input_features = None):
		if input_features is None:
			input_features = [f"x{i}" for i in range(self.phis[0].dim)]
	
		output = []
		for phi in self.phis:
			if np.sum(phi.degree) == 0:
				name = '1'
			else:
				args = []
				for n, p in zip(input_features, phi.degree):
					if p == 1:
						args.append(f'{n:s}')
					elif p > 1:
						args.append(f'{n:s}^{p}')
				
				name = '*'.join(args)
			output += [ name]

		return output

def sindy(phis, Ys, dts, threshold = 1e-2, alpha = 0.05):
	r""" A simple wrapper around PySINDy's implementation of SINDy

	Notes
	-----
	* This uses a fixed, centered finite-difference (second order; equivalent to points=3)
	* Optimization is performed with sequentially thresholded least squares

	Parameters
	----------
	threshold: float
		Small coefficients below this threshold are removed
	alpha: float
		Regularization parameter

	"""
	model = ps.SINDy(
		optimizer = ps.STLSQ(threshold = threshold, alpha = alpha),
		feature_library = Library(phis),
		differentiation_method = ps.FiniteDifference(order = 2, d = 1, is_uniform = True),
	)

	if len(Ys) == 1 and len(dts) == 1:
		model.fit(Ys[0].T, t = dts[0])
	else:
		raise NotImplementedError
		# NB: There is a multiple_trjactories flag
		
	return model.coefficients()
