import numpy as np
import scipy.integrate

################################################################################
# Forward evolution
################################################################################

def evolve_system(phis, C, x0, ts):
	def ode(t, x):
		xp = np.zeros_like(x)
		for phi, c in zip(phis, C.T):
			xp += c * phi.eval(x.reshape(-1,1))
		return xp

	sol = scipy.integrate.solve_ivp(
		ode, 					 	# Callback for derivative
		#(np.min(ts), np.max(ts)),	# Time interval
		(ts[0], ts[-1]),			# Time interval
		x0,							# initial condition
		t_eval = ts,				# times to evaluate at
		method = 'DOP853',			# ODE integrator
		rtol = 1e-6,
		atol = 1e-6,
		)
	if not sol.y.shape[1] == len(ts):
		print(sol)
		assert False
	return sol.y
