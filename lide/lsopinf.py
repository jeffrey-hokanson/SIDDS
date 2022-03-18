import numpy as np
from .matrices import *
import scipy.sparse, scipy.sparse.linalg, scipy.linalg, scipy.optimize
from collections.abc import Sequence

def _format_dts(dts):
    if isinstance(dts, (Sequence, np.ndarray)):
        dts = np.array(dts)
    else:
        dts = np.ones(len(Ys))*dts
    return dts




def lsopinf_grad(phis, Xs, dts, points = 9, mask = None, square = True):
	r""" Compute the gradient with respect to data
	"""

	if len(Xs) > 1: raise NotImplementedError

	X, dt = Xs[0], dts[0]
	d = phis[0].dim
	d, m = X.shape
	n = len(phis)

	if mask is not None:
		flat_mask = vec(mask)	
	else:
		flat_mask = np.ones(d * len(phis), dtype = bool)


	if square:
		D, M = make_DM_square(m, points)
	else:
		D, M = make_DM(m, points)

	Phi = make_Phi(X, phis)
	vec_D = 1/dt * make_vec(D, d).A
	vec_M = make_vec(M, d).A
	vec_Phi = vec_M @ make_vec(Phi, d).A[:,flat_mask]
	vec_x = vec(X)


	U, s, VT = np.linalg.svd(vec_Phi, full_matrices = False)
	V = VT.T
	vec_Phi_pinv = V @ (np.diag(1./s) @ U.T)
	
	def makeC(c):
		flat_C = np.zeros(d*n)
		flat_C[flat_mask] = c
		return unvec(flat_C, d)
	

	# The first term in the derivative: Phi^+ D
	A = vec_Phi_pinv @ vec_D
	
	# The effective coefficients when computing gradient
	vec_Dx = vec_D @ vec_x

	# First term: -Phi^+ [d Phi] Phi^+
	c1 = vec_Phi_pinv @ vec_Dx
	dPhi1 = vec_M @ make_grad_vec_Phi(X, phis, makeC(c1))
	A += -vec_Phi_pinv @ dPhi1

	#vec2 = vec_Dx - vec_Phi @ (vec_Phi_pinv @ vec_Dx)
	vec2 = vec_Dx - U @ (U.T @ vec_Dx)
	vec3 = vec_Phi_pinv.T @ ( vec_Phi_pinv @ vec_Dx)
	for i in range(vec_Phi.shape[1]):
		ei = np.zeros(vec_Phi.shape[1])
		ei[i] = 1
		# Second term: [ Phi^+ Phi^+T [ d Phi^T] (I - Phi Phi^+) ] D x
		# = x.T D.T (I - Phi Phi^+) [d Phi] [ Phi^+ Phi^+T] ]
		c2 = vec_Phi_pinv @ (vec_Phi_pinv.T @ ei)
		dPhi2 = vec2.T @ vec_M @ make_grad_vec_Phi(X, phis, makeC(c2))
		A[i] += dPhi2

		# Third term	
		c3 = ei - V @ (V.T @ ei)
		dPhi3 = vec3.T @ vec_M @ make_grad_vec_Phi(X, phis, makeC(c3))
		A[i] += dPhi3
	
	return A	

def lsopinf(phis, Ys, dts, points = 9, C_true = None, verbose = False, square = True, mask = None):
	r"""Least squares operator inference
	""" 
	
	dts = _format_dts(dts)
	dim = phis[0].dim
	As = []
	bs = []
	for Y, dt in zip(Ys, dts):
		if square:
			D, M = make_DM_square(Y.shape[1], points)
		else:
			D, M = make_DM(Y.shape[1], points)
			
		D *= 1./dt	# Scale for time step

		Phi = make_Phi(Y, phis)
		As += [ M @ Phi ]
		bs += [ D @ Y.T]

	A = np.vstack(As)
	b = np.vstack(bs)

	if mask is not None:
		vecC = np.zeros(dim*len(phis))
		flat_mask = vec(mask)
		vecA = make_vec(A, dim, fmt = 'dense')
		vecb = b.flatten('C')
	
		x, res, rank, s = scipy.linalg.lstsq(vecA[:,flat_mask], vecb)
		vecC[flat_mask] = x
		C = vecC.reshape(-1, dim).T
	else:
		x, res, rank, s = scipy.linalg.lstsq(A, b)
		C = x.T
	
	return C

def lsopinf_regularize(Ys, phis, dts, maxiter = 200, points = 9, tol = 1e-10, q = 0, 
	epsilon0 = 1, C_true = None, rho = 1, alpha = 0.9, verbose = False, rank_est = None):
	dts = _format_dts(dts)
	dim = phis[0].dim	
	As = []
	bs = []
	for Y, dt in zip(Ys, dts):
		D, M = make_DM(Y.shape[1], points)
		D *= 1./dt	# Scale for time step

		Phi = make_Phi(Y, phis)
		As += [ M @ Phi ]
		bs += [ D @ Y.T]
	
	M = len(phis)
	dim = phis[0].dim
	AA = np.vstack(As)
	bb = np.vstack(bs)
	
	A = np.vstack([np.zeros((M*dim, M*dim))] + [np.kron(AA, np.eye(dim))])
	b = np.hstack([np.zeros((M*dim))] + [bb.flatten('C')])
	
	weight = 0*np.zeros((dim, M)) 
	C_old = np.zeros((dim, M))
	if rank_est is None:
		rank_est = (dim*M)//2
	
	for it in range(maxiter):
		A[:dim*M,:] = rho*np.diag(np.sqrt(weight.flatten('C')))
		x, res, rank, s = scipy.linalg.lstsq(A, b)

		C = unvec(x, dim)
		
		delta_C = np.linalg.norm(C - C_old, 'fro')	
	
		mess = f"it {it:4d}\t delta_C {delta_C:5e}\t  eps {epsilon:5e}"
		if C_true is not None:
			mess += f"\t error from truth {np.linalg.norm(C - C_true,'fro'):5e}"
		if verbose:
			print(mess)

		if delta_C < tol or epsilon <= 1e-8:
			break
		C_old = C

		c_mag = np.sort(np.abs(C.flatten()))[::-1]
		epsilon = min(epsilon, alpha * c_mag[rank_est])
		weight = np.abs( C**2 + epsilon)**(q/2-1)
	return C	
