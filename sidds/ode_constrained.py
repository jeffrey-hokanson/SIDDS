import numpy as np
import torch
import torch.nn as nn

from torchdiffeq import odeint
from iterprinter import *


class ODE(nn.Module):
	def __init__(self, phis, C0 = None):
		super(ODE, self).__init__()
		self.phis = phis
		M = len(phis)
		n = phis[0].dim

		C = 0.01*torch.randn((n, M))
		self.C = nn.Parameter(C, requires_grad = True)
		self.x0 = nn.Parameter(torch.randn((n,), requires_grad = True)

	def forward(t, y):
		yp = torch.zeros_like(y)
		for phi in self.phis:
			term = torch.ones(y.shape[0:-1], dtype = y.dtype)
			for j, p in enumerate(phi.degree):
				term *= y[j]**p
			yp += torch.einsum('...,j->...j', term, self.C[:,i])
		return yp
		
			 

def ode_constrained(Ys, phis, C0, x0s, dts, verbose = True, maxiter = 100):
	
	loss_fun = torch.nn.MSELoss(reduction = 'sum')
	optimizer = torch.optim.Adam(model.parameters())
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
	
	for it in range(maxiter):
		pass
