import numpy as np
from lide.denoise import *

import matplotlib.pyplot as plt

def test_finite_diff_matrix(n = 1000, points = 5, deriv = 2):
	t = np.linspace(-1,1, n)
	dt = t[1] - t[0]
	D = finite_diff_matrix(n, deriv = deriv, points = points) 

	f = np.sin(t)
	if deriv % 4 == 1:
		df = np.cos(t)
	elif deriv % 4 == 2:
		df = -np.sin(t)
	elif deriv % 4 == 3:
		df = -np.cos(t)
	elif deriv % 4 == 0:
		df = np.sin(t)

	df_est = (D/dt**deriv) @ f
	err = np.max(np.abs(df - df_est))
	print(err)
	assert err < 1e-5


def plot_l_curve(n = 1000):
	t = np.linspace(-10,10, n)
	dt = t[1] - t[0]
	x = np.sin(t)
	
	y = x + 1e0*np.random.randn(*x.shape)

	D = finite_diff_matrix(n, deriv = 2, points = 3) / dt**2
	lam = np.logspace(-10, 2, 100)
	obj = np.zeros_like(lam)
	reg = np.zeros_like(lam)

	for i, lam_i in enumerate(lam):
		A = scipy.sparse.eye(n) + lam_i * D.T @ D
		lu = scipy.sparse.linalg.splu(A.tocsc())
		yrec = lu.solve(y)
		obj[i] = np.linalg.norm(yrec - y)/np.linalg.norm(y)
		reg[i] = np.linalg.norm(D @ yrec)
		
	fig, ax = plt.subplots()
	sc = ax.scatter(obj, reg, c = np.log10(lam))
	ax.set_xlabel('objective')
	ax.set_ylabel('regularization')
	ax.set_xscale('log')
	ax.set_yscale('log')
	fig.colorbar(sc)
	plt.show()


def	plot_denoise_fixed(n = 1000):
	t = np.linspace(-10,10, n)
	dt = t[1] - t[0]
	x = np.sin(t)
	y = x + 1e0*np.random.randn(*x.shape)

	yrec = denoise_fixed(y.reshape(1,-1), dt, lam = 1e-1).flatten()
	fig, ax = plt.subplots()
	ax.plot(t, x, 'b')
	ax.plot(t, y, 'k', alpha = 0.1)
	ax.plot(t, yrec, 'r', alpha = 0.3)
	plt.show()
	

def test_denoise_corner(n = 1000):
	t = np.linspace(-10,10, n)
	dt = t[1] - t[0]
	X = np.vstack([np.sin(t), np.cos(2*t)])
	
	Y = X + 1e-1*np.random.randn(*X.shape)

	Yrec = denoise_corner(Y, dt)
	
	fig, axes = plt.subplots(2)
	for i, ax in enumerate(axes):
		ax.plot(t, X[i], 'b')
		ax.plot(t, Y[i], 'k', alpha = 0.1)
		ax.plot(t, Yrec[i], 'r', alpha = 0.3)
	plt.show()



if __name__ == '__main__':
	#test_finite_diff_matrix()
	#plot_l_curve()
	#plot_denoise_fixed()
	test_denoise_corner()
