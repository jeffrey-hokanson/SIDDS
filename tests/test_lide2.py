import numpy as np
import scipy.sparse
from lide  import *
import scipy.signal
from util import *
from checkjac import *
import matplotlib.pyplot as plt

def test_cholesky_inverse():
	np.random.seed(0)
	n = 10
	A = np.random.randn(n,n)
	A = A.T @ A + 1e-8* np.eye(n)
	
	M = CholeskyInverse(A)
	err = M @ A - np.eye(n)
	max_err = np.max(np.abs(err))
	print("error", max_err)
	assert max_err < 1e-10

def test_sparse_lu_inverse():
	n = 100
	A = scipy.sparse.diags([1, -2, 1], [-1,0,1], shape = (n, n))
	M = SparseLUInverse(A)
	err = M @ A.toarray() - np.eye(n)
	max_err = np.max(np.abs(err))
	print("error", max_err)
	assert max_err < 1e-10	

def test_build_matrices():
	np.random.seed(0)
	Xs, phis, dts, C = generate_data(M=100, T = 10, noise = 0)
	Ns = [1e-7*np.random.randn(*X.shape) for X in Xs]
	Ys = [X + N for X, N in zip(Xs, Ns)]

	A, B, b, d, Sigma = build_matrices(C, phis, Xs, Ys, dts, points = 9)

	rho = 1e-3
	W = 0e-8*np.eye( C.shape[0]*C.shape[1])

	Sigmas = [scipy.sparse.eye(X.shape[0]*X.shape[1]) for X in Xs]
	points = 9
	dC, dXs, z = kkt_step(W, C, phis, Ys, Xs, dts, points, Sigmas, rho, verbose = True, tol = 1e-8, dense = False)
	print(dC)
	for dX, N in zip(dXs, Ns):
		print(dX/N)



def test_relaxation():
	Xs, phis, dts, C = generate_data(M=1000, T = 1, noise = 0, dt = 1e-5)
	
	Ns = [1e-7*np.random.randn(*X.shape) for X in Xs]

	Ys = [X + N for X, N in zip(Xs, Ns)]
	NC = 1e-3*np.random.randn(*C.shape)
	Ct = C + NC
	points = 7
	
	lide = LIDE(phis, Ys, dts, points = points)
	con = LIDENonlinearEqualityConstraint(phis, Xs, dts, points)
	x = encode(Ct, Xs)
	h = con(x)
	A = con.jac(x)
		
	# If we are given exact data, p_C should be in the same direction
	# as the pertubation
	p_C, p_Xs = lide.relaxation(h, A)
	print(p_C)
	print(-NC)
	
	p = encode(p_C, p_Xs)
	norm_h = np.linalg.norm(h)
	norm_hAp = np.linalg.norm(h + A @ p)
	print("norm h", norm_h, "norm h + A @ p", norm_hAp)
	assert norm_h >= norm_hAp	
	#assert np.allclose(p_C, -NC, rtol =  1e-5, atol = 1e-5)
	
	# If we have noisy data, but exact coefficients, 
	# this step should point towards correct X
	p_C, p_Xs = relaxation(C, phis, Ys, dts, points)
	
	x = encode(C, Ys)
	h = con(x)
	A = con.jac(x)
	p = encode(p_C, p_Xs)
	norm_h = np.linalg.norm(h)
	norm_hAp = np.linalg.norm(h + A @ p)
	print("norm h", norm_h, "norm h + A @ p", norm_hAp)

	#print("p")
	#print(p_Xs[0])
	#print("N")
	#print(Ns[0])
	
	#print("ratio")
	#print(p_Xs[0]/Ns[0])


def test_encode():
	Xs, phis, dts, C = generate_data(M=100, T = 10, noise = 0, dt = 1e-3)
	dims = [X.shape for X in Xs]
	x = encode(C, Xs)
	C_, Xs_ = decode(x, dims)
	assert np.allclose(C, C_)
	assert all([np.allclose(X, X_) for X, X_ in zip(Xs, Xs_)])


def test_constraint():
	Xs, phis, dts, C = generate_data(M=10, T = 1, noise = 0, dt = 1e-2)
	points = 5
	con = LIDENonlinearEqualityConstraint(phis, Xs, dts, points)
	x0 = encode(C, Xs)
	x0 += 1e-3*np.random.randn(*x0.shape)
	h = con.fun(x0)
	A = con.jac(x0)
	res = con.fun
	jac = lambda x: con.jac(x).toarray()
	err = check_jacobian(x0, res, jac)
	print(err)
	assert err < 1e-4
	#print(x0.shape, h.shape, A.shape)


def test_objective():
	Xs, phis, dts, C = generate_data(M=10, T = 1, noise = 0, dt = 1e-2)
	points = 5
	Sigmas = [scipy.sparse.eye(X.shape[0]*X.shape[1]) for X in Xs]
	W = scipy.sparse.eye(C.shape[0]*C.shape[1])
	mu = 1e-3
	Ys = [X + 1e-5*np.random.randn(*X.shape) for X in Xs]
	Ct = C + 1e-1*np.random.randn(*C.shape)
	obj = LIDEObjective(phis, Ys, Sigmas, W, dts, points, mu)
	
	x0 = encode(Ct, Xs)
	print(obj.fun(x0))
	print(obj.jac(x0))

	err = check_grad(obj.fun, obj.jac, x0)
	print("error", err)
	assert err < 5e-5

	err = check_grad(obj.jac, lambda x: obj.hess(x).toarray(), x0)
	print(err)	


def test_solve():
	#Xs, phis, dts, C = generate_data(fun = duffing, M = 2000, T = 1, noise = 0, dt = 1e-2)
	Xs, phis, dts, C = generate_data(fun = vanderpol, M = 1000, T = 1, noise = 0, dt = 1e-2)
	points = 9
	if False:
		import matplotlib.pyplot as plt
		fig = plt.figure()
		ax = plt.axes(projection = '3d')
		ax.plot3D(Xs[0][0], Xs[0][1],Xs[0][2])
		plt.show()
		assert False
	Sigmas = [scipy.sparse.eye(X.shape[0]*X.shape[1]) for X in Xs]
	#np.random.seed(int(time.time()))
	np.random.seed(0)
	Ys = [X + 1e-1*np.random.randn(*X.shape) for X in Xs]
	d = np.max(np.abs(Xs[0]), axis = 1)
	Yfs = [scipy.signal.savgol_filter(Y, window_length = 51, polyorder = 2, axis = 1) for Y in Ys]
	if False:
		import matplotlib.pyplot as plt
		plt.plot(Ys[0].T)
		plt.plot(Yfs[0].T)
		plt.show()

	lide = LIDE(phis, Ys, dts, points = points, mu = 1e2, kkt_reg = 1e-4, 
		solver = 'minres', v_soc = -1, irls = True, square = True, verbose = True, tol_dx = 1e-8, tol_opt = 1e-8,
		stab_coef = 1e-4,
		solve_qp_kwargs = {'maxiter': 1000, 'tol': 1e-10},)
	#lide.solve()
	lide.solve_irls(q = 0 , alpha = 1e-2)
	print("IRLS C")
	print(lide.C)

	print("regularization", lide.regularization())	
	nnz = int(lide.regularization())
	threshold = np.sort(np.abs(lide.C.flatten()))[-nnz]
	mask = np.abs(lide.C) >= threshold
	print('nnz', nnz)
	print(np.sort(np.abs(lide.C.flatten())))
	print(threshold)
	print("IRLS error", np.linalg.norm(lide.C - C, 'fro'))
	lide.tol_opt = 1e-8	
	lide.solve(C0 = lide.C * mask, mask = mask, maxiter = 100)
	print("Masked C")
	print(lide.C)
	print("masked error", np.linalg.norm(lide.C - C, 'fro'))
	#lide.solve(Xs0 = Ys, C0 = None, maxiter =100)
	print("True C")
	print(C)	

def test_oversample():
	#Xs, phis, dts, C = generate_data(fun = duffing, M = 2000, T = 1, noise = 0, dt = 1e-2)
	Xs, phis, dts, C = generate_data(fun = lorenz63, M = 1000, T = 1, noise = 0, dt = 1e-2)
	points = 3
	np.random.seed(0)
	Ys = [X + 5*np.random.randn(*X.shape) for X in Xs]

	lide = LIDE(phis, Ys, dts, points = points, oversample = 8, verbose = True)
	lide.solve(smooth = 1e-6)
	print("Nomial C")
	print(C)
	print("IRLS C")
	print(lide.C)

def test_L_curve():
	Xs, phis, dts, C = generate_data(fun = rossler, M = 500, T = 1, noise = 0, dt = 1e-2)
	points = 9
	Ys = [X + 1e-2*np.random.randn(*X.shape) for X in Xs]
	
	lide = lide_L_corner(phis, Ys, dts, points = points, maxiter = 10)
	print(lide.C)

	
if __name__ == '__main__':
	#test_cholesky_inverse()
	#test_sparse_lu_inverse()
	#test_build_matrices()
	#test_relaxation(relaxation_split)
	#test_constraint()
	#test_objective()
	test_solve()
	#test_L_curve()
	#test_oversample()
