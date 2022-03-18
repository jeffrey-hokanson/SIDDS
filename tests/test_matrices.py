import numpy as np
import pytest
import lide
import scipy.sparse
from util import check_grad

@pytest.mark.parametrize("points", [3,5,7,9,11])
def test_make_DM(points):
	D, M = lide.make_DM(100, points)
	
	x = np.linspace(-1,1,100)
	dx = x[1] - x[0]
	D /= dx

	# Generate a legendre polynomial as test data
	poly_coef = np.zeros(points)
	poly_coef[-1] = 1
	y = np.polynomial.legendre.legval(x, poly_coef)
	poly_coef_der = np.polynomial.legendre.legder(poly_coef)
	yp = np.polynomial.legendre.legval(x, poly_coef_der)
	print(D @ y - M @ yp)
	assert np.allclose(D @ y, M @ yp)

def test_make_DM_square():
	points = 5
	M = 20
	x = np.linspace(1,1+1, M)
	dx = x[1] - x[0]
	D, M = lide.make_DM_square(M, points)

	print(D[:5,:5].A)	
	print(D[-5:,-5:].A)	

	D /= dx
	f = np.sin(x)
	df = np.cos(x)
	df_est = D @ f
	print(np.abs(df - df_est))
	

def test_make_vec():

	# Setup
	phis, C = lide.duffing()
	dim = C.shape[0]
	nsamp = 10
	print("C", C.shape)
	C = np.random.randn(*C.shape)
	X = np.random.uniform(0,1, size = (2, nsamp))
	Phi = lide.make_Phi(X, phis)
	
	vec_Phi = lide.make_vec(Phi, dim)
	#print(vec_Phi.toarray())
	Phic1 = (vec_Phi @ lide.vec(C))
	# TODO: Why do we need to C style vectorize C here, rather than Fortran?
	# Because blocks correspond to a single dimension
	# flattening C style perserves this ordering
	# TODO: S
	Phic2 = lide.vec(C @ Phi.T)
	print(Phic1 - Phic2)	
	assert np.allclose(Phic1, Phic2)
	

@pytest.mark.parametrize("points", [5])
def test_make_grad_Phi(points):

	# Generate test data
	phis, C = lide.duffing()
	dim = phis[0].dim
	nsamp = 5
	
	np.random.seed(0)
	X = np.random.randn(dim, nsamp) 

	C = np.random.randn(*C.shape)
	
	def Phi_c(x_vec):
		X = lide.unvec(x_vec, dim)
		Phi = lide.make_Phi(X, phis)
		return lide.make_vec(Phi, dim) @ lide.vec(C)

	def grad_Phi_c(x_vec):
		X = lide.unvec(x_vec, dim)
		return lide.make_grad_vec_Phi(X, phis, C).toarray()

		
	err = check_grad(Phi_c, grad_Phi_c, lide.vec(X))
	print("error", err)
	assert err < 1e-7, "Incorrect derivative"

	print(grad_Phi_c(X.flatten()))
		

@pytest.mark.parametrize("dim", [1,2,3,4,5])
def test_vec(dim):
	X = np.random.randn(dim, 10)
	X2 = lide.unvec(lide.vec(X), dim)
	assert np.allclose(X, X2)
	x = np.random.randn(dim*10)
	x2 = lide.vec(lide.unvec(x, dim))
	assert np.allclose(x, x2)


if __name__ == '__main__':
	#test_make_grad_Phi(5)
	#test_make_vec()
	test_make_DM_square()
	pass
