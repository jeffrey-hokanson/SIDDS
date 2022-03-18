import numpy as np
import lide
import pytest

from util import *

def test_lide_step_cvxpy():
	Ys, phis, dts, C = generate_data(M=100, T = 50, noise = 0)

	C_old = C
	Xs_old = [np.copy(Y) for Y in Ys]

	C_new, Xs = lide.lide_step_cvxpy(Ys, phis, C_old, Xs_old, dts = dts, points = 9)
	print(C)
	print(C_new)
	print(np.isclose(C, C_new, rtol = 1e-3, atol = 1e-2))
	assert np.allclose(C_new, C, rtol = 1e-3, atol = 1e-2)

@pytest.mark.parametrize("step", ['cvxpy', 'spsolve', 'pyamg', 'gmres'])
def test_lide(step, noise = 1e-5):
	

	Ys, phis, dts, C = generate_data(noise = noise, T = 1, M = 50000, seed = 0, dt = 1e-2)
	C_ls = lide.lsopinf(Ys, phis, dts)
	print('least squares error', f'{np.linalg.norm(C_ls - C, "fro"):5e}')
	
	C_est, Xs = lide.lide(Ys, phis, dts, points = 9, rho = 1e-5, C_true = C, step = step, verbose = 2)

	print(C)
	print(C_est)	
	assert np.allclose(C_est, C, atol = 5e-1,  rtol = 1e-1) 

	

def test_lide_step_kkt():
	Ys, phis, dts, C = generate_data()

	C_old = C + 1e-1*np.random.randn(*C.shape)
	Xs_old = [np.copy(Y) for Y in Ys]

	C_new, Xs = lide.lide_step_cvxpy(Ys, phis, C_old, Xs_old, dts = dts, points = 9)

	C_new2, Xs2 = lide.lide_step_kkt(Ys, phis, C_old, Xs_old, dts = dts, points = 9)

	print(C_new)
	print(C_new2)
	print('max error', np.max(np.abs(C_new - C_new2)))
	assert np.allclose(C_new, C_new2, atol = 1e-2)
	for X, X2 in zip(Xs, Xs2):
		assert np.allclose(X, X2, atol = 1e-6)

def test_lide_regularize():
	Ys, phis, dts, C = generate_data()
	
	C_est, Ns = lide.lide_regularize(Ys, phis, dts, points = 9, lam = 1e-10, C_true = C, p = 0, maxiter = 20)

	print(C)
	print(C_est)	
	assert np.allclose(C_est, C, atol = 5e-1,  rtol = 1e-1) 


def test_encoding():
	from lide.lide import _encode, _decode
	Ys, phis, dts, C = generate_data(T=5, M = 500)
	
	Xs_shape = [Y.shape for Y in Ys]
	dim = phis[0].dim
	N = len(phis)

	x = _encode(C, Ys)
	C2, Ys2 = _decode(x, dim, N, Xs_shape)
	assert np.allclose(C2, C)
	assert np.allclose(Ys, Ys2)

def test_opt():
	from lide.lide import _encode, _decode, _obj, _jac, _constraint, _constraint_jac
	from lide import vec	

	Xs, phis, dts, C = generate_data(T=2, M = 50, noise = 0, dt = 1e-3)
	Xs_shape = [X.shape for X in Xs]
	Ys = [X + 1e-3*np.random.randn(*X.shape) for X in Xs]
	dim = phis[0].dim
	N = len(phis)

	encode = _encode
	decode = lambda x: _decode(x, dim, N, Xs_shape)
 
	obj = lambda x: _obj(*decode(x), Ys)
	jac = lambda x: np.hstack([vec(J) for J in _jac(*decode(x), Ys)])

	x0 = encode(C, Xs)
	err = check_grad(obj, jac, x0)
	assert err < 1e-5
	
	points = 9
	# Check constraint with nomial data
	constraint = lambda x: _constraint(*decode(x), Ys, dts, phis, points)
	constraint_jac = lambda x: _constraint_jac(*decode(x), Ys, dts, phis, points)


	assert np.allclose(constraint(x0), 0), "Should satisfy the constraint with exact data"

	print("x0", x0.shape)
	print("con", constraint(x0).shape)	
	print("con_jac", constraint_jac(x0).shape)	
	err = check_grad(constraint, lambda x: constraint_jac(x).toarray(), x0)
	print(err)
	assert err < 1e-4 
	
	
def test_lide_opt():
	Ys, phis, dts, C = generate_data(T=1, M = 100, noise = 1e-2, dt = 1e-1)
	C, Xs = lide.lide_opt(Ys, phis, dts, points = 9, verbose = True)
	print(C)
	C2, _ = lide.lide(Ys, phis, dts, points = 9)
	print(C2)
	assert np.allclose(C, C2, atol = 1e-2)


	

if __name__ == '__main__':
#	test_lide_step_cvxpy()
#	test_lide_step_kkt()
	test_lide('gmres', noise = 1e-2)
#	test_lide_regularize()
#	test_lide_step_kkt_nullspace()	
#	test_encoding()
#	test_opt()
#	test_lide_opt()
#	test_lide2()
