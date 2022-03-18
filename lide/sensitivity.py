import numpy as np
import scipy.integrate
from tqdm import tqdm

def sensitivity(phis, C, x0, ts, atol = 1e-6, rtol = 1e-6, progress = False):
	r"""

	Parameters
	----------
	C: np.ndarray
		Size (d, *)	
	x0: np.ndarray
		size (d,)

	Returns
	-------
	Dx0s
		derivatives with respect to initial condition
	D_cvec
		derivative with respect to coefficients c
	"""

	d = len(x0)
	n = C.shape[1]

	if progress:
		pbar = tqdm(total = len(ts))
	else:
		pbar = None

	def ode(t, xvw):
		if pbar:
			pbar.update( np.sum(ts < t) - pbar.n)
		# main ODE
		x = xvw[:d]
		xp = np.zeros_like(x)
	
		# sensitivity wrt initial conditions
		v = xvw[d:d + d*d]
		V = v.reshape(d,d).T
		Vp = np.zeros_like(V)	
	
		# sensitivity wrt C
		w = xvw[d+d*d:]
		W = w.reshape(d*n, d).T
		Wp = np.zeros_like(W)	

		for k, (phi, c) in enumerate(zip(phis, C.T)):
			phi_x = phi.eval(x.reshape(-1,1))
			dphi_x = phi.grad(x.reshape(-1,1)) 
			xp += c * phi_x
			Vp += np.outer(c, dphi_x) @ V
			Wp += np.outer(c, dphi_x) @ W
			Wp[:,d*k:d*(k+1)] += phi_x * np.eye(d)
		
		xvw_p = np.hstack([xp, Vp.flatten('F'), Wp.flatten('F')])
		return xvw_p	

	V0 = np.eye(d)
	W0 = np.zeros((d*n, d))
	xvw0 = np.hstack([x0, V0.flatten('F'), W0.flatten('F')])

	sol = scipy.integrate.solve_ivp(
		ode, 		 	# Callback for derivative
		(ts[0], ts[-1]),		# Time interval
		xvw0,			# initial condition
		t_eval = ts,	# times to evaluate at
		method = 'DOP853', # ODE integrator
		rtol = rtol,
		atol = atol,
		)

	# derivative wrt initial conditions
	dvs = []
	dws = []
	for t, xvw in zip(sol.t, sol.y.T):
		# As sometimes the solver will insert time points other than those desired,
		# here we remove those 
		if np.min(np.abs(t - ts)) <= 1e-10*t:
			dvs += [ xvw[d:d + d*d].reshape(d,d).T]
			dws += [ xvw[d+d*d:].reshape(-1,d).T ]

	if pbar:
		pbar.close()
	return dvs, dws

