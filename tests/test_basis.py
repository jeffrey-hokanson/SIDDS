import numpy as np
from lide.basis import *
from scipy.optimize import check_grad
import pytest

@pytest.mark.parametrize("degree",[
	[0],
	[1],
	[2],
	[0,3],
	[2,3],
	[1,2,3,4]
])
def test_monomial(degree):
	mono = Monomial(degree)
	
	np.random.seed(0)
	for it in range(10):
		x = np.random.randn(len(degree))
		err = check_grad(
			lambda x:mono.eval(x.reshape(len(degree),1)).flatten(), 
			lambda x:mono.grad(x.reshape(len(degree),1)).flatten(),
			x)
		print("error", err)
		assert err < 1e-6


if __name__ == '__main__':
	test_monomial([1,2,3])
