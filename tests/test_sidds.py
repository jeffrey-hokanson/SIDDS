import numpy as np
from lide import *
from lide.sidds_nullspace import *

def test_sidds_constraint():
	M = 100
	phis, C, x0 = simple_harmonic_oscillator(include_constant = False)
	dt = 1e-2
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	
	con = Constraint(phis, [t])
	h = con.fun(con.encode(C, [X]))
	d, n = C.shape
	print(h)

	A = con.jac(con.encode(C, [X])).tolil()
	K = A[:,:d * n+d]
	L = A[:,d*n+d:]
	# Overkill!
	splu = scipy.sparse.linalg.splu(L.tocsc())
	N = splu.solve(K.A)
	print(N.shape)
	U = np.vstack([np.eye(d + n * d), -N])
	U, R = np.linalg.qr(U, mode = 'reduced')
	#print(np.linalg.svd(R))
	print(A @ U)
	#print(N)
	pass


if __name__ == '__main__':
	test_sidds_constraint()
