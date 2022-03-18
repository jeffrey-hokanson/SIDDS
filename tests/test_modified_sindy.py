from lide import *


def test_modified_sindy(M = 1000, dt = 1e-2):
	phis, C, x0 = duffing()
	print("C\n", C.T)
	dim = phis[0].dim
	
	t = dt * np.arange(M)
	X = evolve_system(phis, C, x0, t)
	Xs = [X]
	dts = np.array([dt])

	noise = 1e-2
	np.random.seed(0)
	Ys = [X + noise*np.random.randn(*X.shape) for X in Xs]

	C, Xs = modified_SINDy(3, Ys, dts)
	print(C)
	print(Xs)

if __name__ == '__main__':
	test_modified_sindy()
