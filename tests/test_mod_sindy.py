import numpy as np
import scipy.sparse
from lide  import *
from lide.modified_sindy import *
import scipy.signal
from util import *
from checkjac import *
import matplotlib.pyplot as plt


def test_solve():
	Xs, phis, dts, C = generate_data(fun = lorenz63, M = 2500, T = 1, noise = 0, dt = 1e-2)
	phis = [phi for phi in phis if np.sum(phi.degree) <= 3]
	np.random.seed(0)
	Ys = [X + 1e-2*np.random.randn(*X.shape) for X in Xs]
	Cest, Xs = modified_SINDy(Ys, phis, dts, noise_init = True, q = 1, lam = 0.2)
	print("Finished")
	print(C)
	print(Cest)


if __name__ == '__main__':
	test_solve() 	
