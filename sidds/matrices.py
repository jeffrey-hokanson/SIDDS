import numpy as np
import scipy.sparse, scipy.misc

def make_DM(length, points):
	r""" Construct the vectorized derivative and mass matrices


	Returns
	-------
	D: scipy.sparse.csr
		Differencing matrix.
		*Note* This assumes unit time step; for a time step delta, set D /= delta
	M: scipy.sparse.csr
		Matrix to extract corresponding time point
	"""

	w = scipy.misc.central_diff_weights(points)
	w[ (points-1)//2] = 0.		# Fixes a rounding issue for the center
	offsets = np.arange(points)
	D = scipy.sparse.diags(w, offsets, shape = (length - (points-1), length)).tocsr()
	M = scipy.sparse.diags([1], (points-1)//2, shape = (length - points+1, length)).tocsr()
	return D, M

def make_DM_square(length, points):
	r"""
	"""
	# Generic coefficients
	bw = (points - 1) //2
	x = np.arange(-bw, bw+1, dtype = float)
	V = np.vstack([x**p for p in range(points)])
	b = np.zeros(points)
	b[1] = 1
	lu = scipy.linalg.lu_factor(V)	
	w = scipy.linalg.lu_solve(lu, b)
	#w += scipy.linalg.lu_solve(lu, b - V @ w) # One step of iterative refinement
	w[bw] = 0 # central term shouldn't contribute 
	
	offsets = np.arange(-bw, bw+1, dtype = int)
	diags = [w[i]*np.ones(length - k) for i, k in enumerate(offsets)]
	D = scipy.sparse.diags(diags, offsets, shape = (length, length)).todok()

	# Fix beginning/ending points
	for i in range(bw):
		x = np.arange(-i, points - i, dtype = float)
		V = np.vstack([x**p for p in range(len(x))])
		b = np.zeros(len(x))
		b[1] = 1
		lu = scipy.linalg.lu_factor(V)	
		w = scipy.linalg.lu_solve(lu, b)
		#w += scipy.linalg.lu_solve(lu, b - V @ w) # One step of iterative refinement
		for j in range(len(w)):
			D[i, j] = w[j]
			D[-i-1,-j-1] = -w[j]

	D = D.tocsr()
	# TODO: Replace with identity operator?
	M = scipy.sparse.eye(length)
	return D, M


def make_vec(X, dim, fmt = 'sparse'):
	r"""

	Form the vectorized form with the multiple blocks for each dimension
	
	"""
	assert fmt in ['sparse', 'dense']
	In = scipy.sparse.eye(dim)
	XX = scipy.sparse.kron(X, In)
	if fmt == 'sparse':
		return XX
	else:
		return XX.A

def make_Phi(X, phis):
	Phi = np.zeros((X.shape[1], len(phis)), dtype = X.dtype)
	for k, phi in enumerate(phis):
		#print('X', X.shape, 'Phi', Phi.shape, phi.eval(X).shape)
		Phi[:,k] = phi.eval(X)

	return Phi

def vec(X):
	return X.flatten('F')

def unvec(X, dim):
	return X.reshape(-1, dim).T


def make_grad_vec_Phi(X, phis, C):
	r"""

	Forms 

		\vma M \nabla \vma \Phi(\vve x) \vve c \in ((M-ell)\times n) \times(M \times n)
	
	"""
	dim, nsamp = X.shape
	ii = []
	jj = []
	vv = []

	# Pre-evaluate the gradient of each phi
	grad_phi = np.array([phi.grad(X) for phi in phis])

	for i in range(nsamp): 		# Iterate over different time points
		for j in range(dim):	# iterate over dimensions
			# D_{i*n + j} \Phi is zero execpt for the ith row, 
			DPhi = grad_phi[:,i, j]
			for k in range(dim):
				# Then for the kth block in vec Phi, 
				# we multiply by the corresponding block in the coefficients C
				DPhic = DPhi @ C[k,:]
				ii += [int(k + dim*i)]
				jj += [int(dim*i + j)]
				vv += [float(DPhic)]
	grad_Phi = scipy.sparse.coo_matrix((vv, (ii, jj)), (nsamp*dim, nsamp*dim)).tocsr()

	return grad_Phi


