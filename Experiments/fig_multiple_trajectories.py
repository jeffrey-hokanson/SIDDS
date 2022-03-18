import numpy as np
import matplotlib.pyplot as plt
import os
from pgf import PGF
from polyrat import total_degree_index
from lide import evolve_system, Monomial, LIDE
from streamline import streamline 

def fun_ex3(x):
	r""" Example 3, WX19
	"""
	x1 = x[0]
	x2 = x[1]
	return np.array([ x1*(1-x1 - x2), x2*(0.5 - 0.25*x2 - 0.75*x1)])

def ex3(total_degree = 2, include_constant = True):
	dim = 2
	assert total_degree >= 2
	index = total_degree_index(dim, total_degree)
	if include_constant is False:
		index = index[1:]

	C = np.zeros((dim, len(index)), dtype = float)

	phis = []
	for k, degree in enumerate(index):
		phis.append(Monomial(degree))
		if np.array_equal(degree, [1,0]):
			C[0,k] = 1
		if np.array_equal(degree, [2,0]):
			C[0,k] = -1
		if np.array_equal(degree, [1,1]):
			C[0,k] = -1
		
		if np.array_equal(degree, [0,1]):
			C[1,k] = 0.5
		if np.array_equal(degree, [0,2]):
			C[1,k] = -0.25
		if np.array_equal(degree, [1,1]):
			C[1,k] = -0.75
		

	x0 = np.array([1,0])
	return phis, C, x0
	

def stream_plot(f, path, ax = None):
	x1 = np.linspace(-1,2, 100)
	x2 = np.linspace(-0.5, 3, 100)
	X1, X2 = np.meshgrid(x1, x2)

	xx = np.vstack([X1.flatten(), X2.flatten()])
	dxx = f(xx)
	dX1 = dxx[0].reshape(X1.shape)
	dX2 = dxx[1].reshape(X2.shape)
	if ax is None:
		fig, ax = plt.subplots()

	sp = ax.streamplot(X1, X2 ,dX1, dX2, density = 1.3)

	lines = [] # The collection of streamlines
	line = [] # The line we are currently building
	for i, l in enumerate(sp.lines.get_segments()):
		# Annoyingly, these segments are all length 2, so we join them together
		assert l.shape == (2,2)
		if len(line) == 0:
			line = [l[0], l[1]]
		else:
			if np.array_equal(line[-1], l[0]):
				if not np.array_equal(line[-1], l[1]):
				#if np.linalg.norm(line[-1] - l[1]) > 1e-6:
					line += [l[1]]
			else:
				lines += [np.array(line)]
				line = []

	# make the directory
	try:
		os.mkdir(path)
	except  FileExistsError:
		pass

	for i, l in enumerate(lines):
		li = list(l)
		j = 1
		# Truncate duplicate
		while True:
			try:
				if np.linalg.norm(li[j-1] - li[j]) > 1e-5:
					j += 1
				else:
					li.pop(i)
			except IndexError:
				break
		l = np.vstack(li)
		print(l.shape)
		pgf = PGF()	
		pgf.add('x', l[:,0])
		pgf.add('y', l[:,1])
		pgf.write(os.path.join(path, f'line_{i:d}.dat'))
	
	return lines


def stream_plot(fun, path, ax = None):
	xrange = (-1, 2)
	yrange = (-0.5, 3)
	print(fun(np.array([-1, -0.5])))
	lines = streamline(fun, xrange, yrange, num_lines = 70)
	
	for i, l in enumerate(lines):
		li = list(l)
		j = 1
		# Truncate duplicate
		while True:
			try:
				if np.linalg.norm(li[j-1] - li[j]) > 1e-3:
					j += 1
				else:
					li.pop(j)
			except IndexError:
				break
		l = np.vstack(li)
		pgf = PGF()	
		pgf.add('x', l[:,0])
		pgf.add('y', l[:,1])
		pgf.write(os.path.join(path, f'line_{i:d}.dat'))
	
	return lines

def fit_lide(phis, Ys, dts):
	lide = LIDE(phis, Ys, dts, verbose = True)
	lide.solve()
	return lide.C

def fun_vec(phis, C):
	def ode(x):
		x = np.atleast_2d(x).T
		xp = np.zeros_like(x)
		for phi, c in zip(phis, C.T):
			xp += np.outer(c, phi.eval(x))
		return xp

	return ode
	

if __name__ == '__main__':
	phis, C, x0 = ex3()

	dt = 1e-1
	t = dt*np.arange(100)
	
	x0s = [
		[0, 0.1],
		[0.1, 0],
#		[0.25, 0.25],
		[-0.5, 1],
	]
	
	Xs = [evolve_system(phis, C, x0, t) for x0 in x0s]
	print(Xs)

	dts = [dt,]*len(Xs)

	for i in range(len(Xs)):
		fig, ax = plt.subplots()
		Xi = Xs[i]
		ax.plot(Xi[0], Xi[1], 'r')
		C0 = fit_lide(phis, Xs[:i+1], dts[:i+1])
		print(C0)
		fun0 = fun_vec(phis, C0)
		stream_plot(fun0, f'data/fig_multiple_trajectories{i:d}', ax = ax)
		pgf = PGF()
		pgf.add('x', Xi[0])
		pgf.add('y', Xi[1])
		pgf.write(f'data/fig_multiple_trajectories_path{i:d}.dat')
	
		
	print(C0)
	stream_plot(fun_vec(phis, C), f'data/fig_multiple_trajectories_truth')

	plt.show()
	
