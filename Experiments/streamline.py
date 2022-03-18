import numpy as np
from scipy.integrate import solve_ivp, ode
from scipy.spatial import Voronoi, KDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import find_objects, label
import matplotlib.pyplot as plt
import random

def _unit_vecfield(fun, backwards = False):
	r""" Generate a version of the vector field with unit length

	Parameters
	----------
	fun: function
		A function of a single argument returning the gradient of the vector field
	
	Returns
	-------
	fun_wrapper: function
		A function of (t, x) -> x used for evolving a differential equation

	"""
	if backwards:
		def fun_wrapper(t, x):
			dx = -fun(x)
			scale = (1e-4 + np.linalg.norm(dx))
			return dx/scale
	else:
		def fun_wrapper(t, x):
			dx = fun(x)
			scale = (1e-4 + np.linalg.norm(dx))
			return dx/scale

	return fun_wrapper

def _evolve(fun, x0, dt, xrange, yrange):
	r""" Return the trajectory inside the domain passing through x0
	"""
	ufun = _unit_vecfield(fun)

	def isinside(xc):
		xc1, xc2 = xc
		if not (xrange[0] <= xc1 <= xrange[1]):
			return False
		if not (yrange[0] <= xc2 <= yrange[1]):
			return False
		return True	

	odeint = ode(ufun).set_integrator('lsoda')
	# Evolve foward
	odeint.set_initial_value(x0, 0)
	xf = [x0]
	while odeint.successful():
		odeint.integrate(odeint.t + dt)
		xf.append(odeint.y)

		# Check if we've reached a sink 
		# by looking for movement
		delta_x = np.linalg.norm(xf[-1] - xf[-2])
		if delta_x < 1e-1*dt:
			break

		# Check if we've escaped the bounding box
		if not isinside(xf[-1]):
			xend = xf.pop(-1)
			odeint.set_initial_value(xf[-1], 0)
			# how much we overshot the box
			d = _dist_box(xend, xrange, yrange)
			# Essentially perform one step of Newton's method to correct
			odeint.integrate(max(dt - d,0))
			xf.append(odeint.y)
			break

	# Evolve backwards
	ufun = _unit_vecfield(fun, backwards = True)
	odeint = ode(ufun).set_integrator('lsoda')
	odeint.set_initial_value(x0, 0)
	xb = [x0]
	while odeint.successful():
		odeint.integrate(odeint.t + dt)
		xb.append(odeint.y) 

		# Check if we've reached a sink 
		# by looking for movement
		delta_x = np.linalg.norm(xb[-1] - xb[-2])
		if delta_x < 1e-1*dt:
			break

		# Check if we've escaped the bounding box
		if not isinside(xb[-1]):
			xend = xb.pop(-1)
			odeint.set_initial_value(xb[-1], 0)
			# how much we overshot the box
			d = _dist_box(xend, xrange, yrange)
			# Essentially perform one step of Newton's method to correct
			odeint.integrate(max(dt - d,0))
			xb.append(odeint.y)
			break

	# Remove the first entry so we don't duplicate
	xx = list(reversed(xb[1:])) + xf
	return np.vstack(xx)

def _isinside(X, xrange, yrange, tol = 0):
	I = (xrange[0] - tol <= X[:,0]) & (X[:,0] <= xrange[1] + tol) 
	I = I & (yrange[0] - tol <= X[:,1]) & (X[:,1] <= yrange[1] + tol)
	return I
	
def _find_empty(traj, xrange, yrange, n_return = 10):
	r""" Find a hole in the trajectory plot
	"""
	X = np.vstack(traj)
	I = _isinside(X, xrange, yrange, tol = 1e-10)
	X = X[I]
	# Mirror points across each axis
	XN, XS, XW, XE = np.copy(X), np.copy(X), np.copy(X), np.copy(X)
	XN[:,1] = yrange[1] + (yrange[1] - X[:,1])
	XE[:,0] = xrange[1] + (xrange[1] - X[:,0])
	XS[:,1] = yrange[0] - (X[:,1] - yrange[0])
	XW[:,0] = xrange[0] - (X[:,0] - xrange[0])

	XX = np.vstack([XN, XS, XW, XE, X])
	vor = Voronoi(XX)
	V = vor.vertices
	tol = 1e-7
	Vint = V[_isinside(V, xrange, yrange, tol  =1e-7)]
	if False:
		ax.plot(Vint[:,0], Vint[:,1], 'r+')

	d = np.min(cdist(Vint, X), axis = 1)
	I = np.argsort(-d)
	return Vint[I[:n_return]], d[I[:n_return]]



def _find_empty_grid(traj, xrange, yrange, ngrid = 100, n_return = 10):
	r""" Switch to finding an empty grid cell
	"""
	x = np.linspace(*xrange, int(ngrid))
	y = np.linspace(*yrange, int(ngrid))

	X, Y = np.meshgrid(x, y)
	Xgrid = np.vstack([X.flatten(), Y.flatten()]).T
	X = np.vstack(traj)
	tree = KDTree(X)
	d, _ = tree.query(Xgrid)
	I = np.argsort(-d)
	return Xgrid[I[:n_return]], d[I[:n_return]]

def _find_empty_rand(traj, xrange, yrange, N = 1000):
	r""" Switch to finding an empty grid cell
	"""
	x = np.random.uniform(*xrange, size = N)
	y = np.random.uniform(*yrange, size = N)
	Xtest = np.vstack([x,y]).T
	X = np.vstack(traj)
	d = np.min(cdist(Xtest, X), axis = 1)
	i = np.argmax(d)
	return Xtest[i], d[i]


def _traj_dist(X, Y):
	return np.max(np.min(cdist(X, Y), axis = 1))

def _dist_box(x, xrange, yrange):
	dx = max(xrange[0] - x[0], 0, x[0] - xrange[1])
	dy = max(yrange[0] - x[1], 0, x[1] - yrange[1])
	return np.sqrt( dx * dx + dy * dy)


def _max_contiguous_subset(seq):
	r"""

	https://stackoverflow.com/a/21690865/3597894
	"""
	i = this_sum = max_sum = 0
	start_idx, end_idx = 0, -1
	for j in range(len(seq)):
		this_sum += seq[j]
		if this_sum > max_sum:
			max_sum = this_sum
			start_idx = i
			end_idx = j
		elif this_sum < 0:
			this_sum = 0
			i = j+1
	return slice(start_idx, end_idx)

def _trim(x, traj, tol = 1e-5):
	X = np.vstack(traj)
	d = np.min(cdist(x, X), axis = 1)
	I = d >= tol
	# find the first contiguous block
	regions = find_objects(label(I)[0])
	size = [np.sum(I[r]) for r in regions]
	I = regions[np.argmax(size)]
	return x[I]
	

def _traj_mean_distance(x, y):
	r"""
	"""
	tree = KDTree(y)
	d, _ = tree.query(x)
	return np.mean(d)

def streamline(fun, xrange, yrange, num_lines = 100, maxiter = 100, dt = None):
	r"""

	"""
	if dt is None:
		scale = min(xrange[1]-xrange[0], yrange[1] - yrange[0])
		dt = 1e-3*scale

	# Deterministically start in the center
	x0 = np.array([np.mean(xrange), np.mean(yrange)])
	traj = [_evolve(fun, x0, dt, xrange, yrange)]
	
	for it in range(num_lines-1):
		x0s, _ = _find_empty_grid(traj, xrange, yrange)
		dists = []
		xs = []
		for i, x0 in enumerate(x0s):
			x = _evolve(fun, x0, dt, xrange, yrange)
			dists.append(_traj_mean_distance(x, np.vstack(traj)))
			xs.append(x)
			#print(i, f'[{x0[0]:+5e} {x0[1]:+5e}] dist {dists[-1]:5e}',)
		
		i = np.argmax(dists)
		traj.append(xs[i])
		print(it, f'dist {dists[i]:5e}',)


	return traj

	for it in range(10):
		dists = [
			_traj_mean_distance(traj[i], np.vstack(traj[:i] + traj[i+1:])) 
			for i in range(len(traj))]
		i = np.argmin(dists)
		if i == num_lines - 1 and it > 0:
			break 
		print(i, dists[i])
		traj.pop(i)
		x0s, _ = _find_empty_grid(traj, xrange, yrange)
		dists = []
		xs = []
		for i, x0 in enumerate(x0s):
			x = _evolve(fun, x0, dt, xrange, yrange)
			dists.append(_traj_mean_distance(x, np.vstack(traj)))
			xs.append(x)
		
		i = np.argmax(dists)
		traj.append(xs[i])
		print(i, dists[i])
	
	fig, ax = plt.subplots()
	for x in traj:
		ax.plot(x[:,0], x[:,1])
	import matplotlib.patches as patches
	rect = patches.Rectangle((xrange[0], yrange[0]), xrange[1]-xrange[0], yrange[1]-yrange[0], fill= False)
	ax.add_patch(rect)
	plt.show()

	return 
	# shuffle the list of trajectories in place
	#random.shuffle(traj)
	for it in range(0*num_lines):
		traj_old = traj.pop(0)
		x0, dist = _find_empty_grid(traj, xrange, yrange, dt)
		x = _evolve(fun, x0, dt, xrange, yrange)
		#dist = _traj_dist(traj_old, traj_new)
		print("it", it, "removed", dist)
		traj.append(x)
		#traj.append(_trim(traj_new, traj, tol = 5e-3*scale))
		


if __name__ == '__main__':

	def fun(x):
		x1 = x[0]
		x2 = x[1]
		return np.array([ x1*(1-x1 - x2), x2*(0.5 - 0.25*x2 - 0.75*x1)])

	xrange = (-1,1)
	yrange = (-0.5, 3)

	streamline(fun, xrange, yrange, maxiter = 0, num_lines = 70)
