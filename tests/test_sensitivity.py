import numpy as np
from lide import evolve_system
from lide.examples import *
from lide.sensitivity import *

def fd_derivative(fun, x, h = 2e-7):
	f = fun(x)
	dx = np.zeros(list(f.shape) + [ len(x)], dtype = float)
	
	for i in range(len(x)):
		ei = np.zeros_like(x)
		ei[i]= 1.		
		f1 = fun(x + ei * h)
		f2 = fun(x - ei * h)
		#print(f1, f2, (f1-f2)/(2*h))
		dx[...,i] = (f1 - f2)/(2*h)
		#print(dx[...,i])

	return dx


def test_sensitivity_long():
	phis, C, x0 = lorenz63()
	n, d = C.shape
	
	ts = 1e-2*np.arange(int(1e5))
	dx0s, dcvecs = sensitivity(phis, C, x0, ts, progress = True, atol = 1e-2, rtol = 1e-4)
	print("Found", len(dx0s), len(dcvecs), "of", len(ts))

def test_sensitivity():
	phis, C, x0 = lorenz63()
	n, d = C.shape

	dt = 1e-2

	print("==========Forward time==========")
	dx0s, dcvecs = sensitivity(phis, C, x0, [0,dt])
	dx0 = dx0s[-1]
	dcvec = dcvecs[-1]

	def forwards_x0(x0):
		X = evolve_system(phis, C, x0, [0,dt,])
		return X[:,-1]
	
	dx0_est = fd_derivative(forwards_x0, x0) 
	print("Finite Difference Derivative")
	print(dx0_est)
	print("Sensitivity Derivative")
	print(dx0)
	assert np.allclose(dx0_est, dx0)
	
	def forwards_cvec(cvec):
		C1 = cvec.reshape(d, n).T
		X = evolve_system(phis, C1, x0, [0,dt,])
		return X[:,-1]
	
	dcvec_est = fd_derivative(forwards_cvec, C.flatten('F')) 
	print("Finite Difference Derivative")
	print(dcvec_est)
	print("Sensitivity Derivative")
	print(dcvec)
	print(C.shape, dcvec_est.shape, dcvec.shape)
	assert np.allclose(dcvec, dcvec_est)
	
	
	#x1 = evolve_system(phis, C, x0, [0, dt])[:,-1]
	#x0ff = evolve_system(phis, C, x1, [0, -dt])[:,-1]
	#print("x0", x0, "x0ff", x0ff)

	print("==========Backwards time==========")
	dx0s, dcvecs = sensitivity(phis, -C, x0, [0,dt])
	dx0 = dx0s[-1]
	dcvec = dcvecs[-1]

	def backwards_x0(x0):
		X = evolve_system(phis, C, x0, [0, -dt])
		return X[:,-1]
	
	dx0_est = fd_derivative(backwards_x0, x0) 
	print("Finite Difference Derivative")
	print(dx0_est)
	print("Sensitivity Derivative")
	print(dx0)
	assert np.allclose(dx0_est, dx0)
	
	def backwards_cvec(cvec):
		C1 = cvec.reshape(d, n).T
		X = evolve_system(phis, C1, x0, [0, -dt])
		return X[:,-1]
	
	
	
	dcvec_est = fd_derivative(backwards_cvec, C.flatten('F')) 
	print("Finite Difference Derivative")
	print(dcvec)
	print("Sensitivity Derivative")
	print(dcvec)
	print(C.shape, dcvec.shape)
	assert np.allclose(dcvec, dcvec_est)

if __name__ == '__main__':
	#test_sensitivity()
	test_sensitivity_long()

	test_sensitivity_long()
