import numpy as np
import lide
import pytest


@pytest.mark.parametrize("fun",[
	lide.duffing,
	lide.lorenz63,
	lide.vanderpol,	
	])
def test_example(fun):
	phis, C = fun()
	dim = phis[0].dim
	assert dim == C.shape[0]
	x0 = np.random.uniform(0,1, size = (dim,) )
	ts = 0.1*np.arange(1000)
	X = lide.evolve_system(phis, C, x0, ts)
	assert X.shape[0] == dim

if __name__ == '__main__':
	test_example(lide.duffing)
