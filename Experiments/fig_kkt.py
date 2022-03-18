import numpy as np
import lide
import scipy.sparse
import scipy.sparse.linalg
from scipy.linalg import solve_triangular
from lide.matrices import *

def build_kkt(Ys, phis, C_old, Xs_old, dts, points = 3, weight = None, x0 = None,
	rho = 1e-4, verbose = False,  **kwargs):
	
	dim = phis[0].dim
	N = len(phis)	

	As = [] # First column in the constraint matrix
	Bs = [] # Diagonal in the constraint matrix
	Gs = []	# block diagonals of the quadratic objective matrix (1,1)-KKT block
	bs = [] # right hand side of constraints
	ds = [] # right hand side for objective 
	
	# The (1,1) block of the KKT matrix corresponding to the penalty on c
	if weight is None:
		W = scipy.sparse.csr_matrix( (dim*N, dim*N))
	else:
		W = scipy.sparse.csr_matrix( (dim*N, dim*N))
	
	for Y, X_old, dt in zip(Ys, Xs_old, dts):
		D, M = make_DM(Y.shape[1], points)
		D *= 1./dt	# Scale for time step
		
		vec_D = make_vec(D, dim)
		vec_M = make_vec(M, dim)	
	
		# Make Phi and its gradient
		Phi = make_Phi(X_old, phis)
		vec_Phi = make_vec(Phi, dim)
		vec_grad_Phi_c = make_grad_vec_Phi(X_old, phis, C_old) 

		A = vec_M @ vec_Phi
		B = vec_M @ vec_grad_Phi_c - vec_D
		b = vec_M @ vec_grad_Phi_c @ vec(X_old)		
		
		As += [A]
		Bs += [B]
		bs += [b]
		ds += [vec(Y)]

		Gs += [scipy.sparse.identity(Y.shape[0]*Y.shape[1])]

	################################################################################	
	# Assemble KKT matrix from blocks 
	################################################################################	
	G = scipy.sparse.block_diag(Gs)
	A = scipy.sparse.vstack(As)
	B = scipy.sparse.block_diag(Bs)
	d = np.hstack(ds)
	b = np.hstack(bs)

	for AA in [G, A, B]:
		AA.eliminate_zeros()

	if rho > 0:	
		KKT_LHS = scipy.sparse.bmat([
			[W + rho * A.T @ A,  rho * A.T @ B,     A.T],
			[rho * B.T @ A,       G + rho * B.T @ B, B.T],
			[A,                      B,                    None]
			])

		KKT_RHS = np.hstack([
			rho * A.T @ b,
			d + rho * B.T @ b,
			b
		])

	else:
		KKT_LHS = scipy.sparse.bmat([
			[W ,  None, A.T ],
			[None, G   , B.T ],
			[A   , B   , None]
			])

		KKT_RHS = np.hstack([
			np.zeros(W.shape[0]),
			d,
			b
		])
		
	KKT_LHS = KKT_LHS.tocsc()

	################################################################################	
	# Generate the initial condition for GMRES
	################################################################################	
	if x0 is None:
		# This should only run on the first iteration; later we'll have KKT data
		x0 = np.hstack([vec(C_old)] + [vec(X) for X in Xs_old])
		x0 = np.hstack([x0, np.zeros(KKT_LHS.shape[0] - len(x0))])

	################################################################################	
	# Build the preconditioner 
	################################################################################	
	
	# Preconditioner for the (1,1) block
	# as this block is dense and SPD we use Cholesky
	b1 = W.shape[0]
	L = scipy.linalg.cholesky(KKT_LHS[:b1,:b1].toarray(), lower = True)
	M1 = scipy.sparse.linalg.LinearOperator( (b1, b1),
		lambda x: solve_triangular(L, solve_triangular(L, x, lower = True), lower = True, trans = 'T')) 

	#print("error block 1", np.linalg.norm(M1 @ KKT_LHS[:b1,:b1].toarray() - np.eye(M1.shape[0])))

	# Preconditioner for the (2,2) block
	# as this block is diagonal with low bandwidth, we use a complete LU factorization
	b2 = b1 + G.shape[0]
	ilu2 = scipy.sparse.linalg.splu(KKT_LHS[b1:b2,b1:b2].tocsc())
	M2 = scipy.sparse.linalg.LinearOperator((b2 - b1, b2 - b1), lambda x: ilu2.solve(x))
	
	# Preconditioner for the (2,2) block
	# as with the previous block, we have a low-bandwith matrix and can use a complete LU
	b3 = KKT_LHS.shape[0]
	if True:
		ilu3 = scipy.sparse.linalg.splu(( scipy.sparse.eye(B.shape[0]) + (B @ B.T)).tocsc())
		M3 = scipy.sparse.linalg.LinearOperator((b3 - b2, b3 - b2), lambda x: ilu3.solve(x))
	else:
		M3 = scipy.sparse.linalg.LinearOperator((b3 - b2, b3 - b2), lambda x: rho*x)
	
	# The total preconditioner	
	M = scipy.sparse.linalg.LinearOperator(KKT_LHS.shape,
		lambda x: np.hstack([M1 @ x[:b1], M2 @ x[b1:b2], M3 @ x[b2:b3]])
		)

	return KKT_LHS, KKT_RHS, M, x0


if __name__ == '__main__':
	from pyamg.krylov import gmres
	from scipy.sparse.linalg import minres 
	from psdr.pgf import PGF

	phis, C = lide.lorenz63()
	dt = 1e-2
	N = 10/dt
	x0 = np.array([-8, 7,27])
	t = dt * np.arange(N)
	X = lide.evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = [dt]
	
	np.random.seed(0)
	Ys = [X + 0.1*np.random.randn(*X.shape) for X in Xs]
	C_old = 0*C

	A, b, M, x0 = build_kkt(Ys, phis, C_old, Ys, dts, points = 9, rho = 1e-4)

	#res = []
	#gmres(A, b, M = M, x0 = x0, tol = 1e-10, maxiter = 1000, residuals = res)
	res = []
	callback = lambda x: res.append(np.linalg.norm(A @ x - b))
	minres(A, b, x0 = x0, M = M, callback = callback, maxiter = 1000, tol = 1e-20)
	for i, r in enumerate(res):
		print(f'{i:4d}\t{r:5e}')
	pgf = PGF()
	pgf.add('residual', res)
	pgf.add('iter', np.arange(len(res)))
	pgf.write('data/fig_kkt_precond.dat')
	
	res = []
	callback = lambda x: res.append(np.linalg.norm(A @ x - b))
	minres(A, b, x0 = x0, callback = callback, maxiter = 1000, tol = 1e-20)
	for i, r in enumerate(res):
		print(f'{i:4d}\t{r:5e}')
	pgf = PGF()
	pgf.add('residual', res)
	pgf.add('iter', np.arange(len(res)))
	pgf.write('data/fig_kkt_normal.dat')
