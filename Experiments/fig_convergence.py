import numpy as np
from lide.examples import lorenz63
from lide import evolve_system, LIDE
from pgf import PGF

phis, C, x0 = lorenz63()
dt = 1e-2
t = dt * np.arange(2000)
X = evolve_system(phis, C, x0, t)
Xs = [X]
dts = [dt]

noise = 1e-1
np.random.seed(0)
Ys = [X + noise * np.random.randn(*X.shape) for X in Xs]

lide = LIDE(phis, Ys, dts, verbose = True)
history = lide.solve_irls(q = 0, alpha = 0.5, history = True)

history2 = lide.solve_polish(history = True)


pgf = PGF()
pgf.add('it', [h['it'] for h in history] + [np.nan] + [h['it'] + history[-1]['it'] + 1 for h in history2] )
pgf.add('con_norm', [h['constraint'] for h in history] + [np.nan] + [h['constraint'] for h in history2 ])
pgf.add('C_err', [np.linalg.norm(C - h['C'], 'fro') for h in history] + [np.nan] + [np.linalg.norm(C - h['C'], 'fro') for h in history2])
pgf.add('X_err', [np.linalg.norm(X - h['Xs'][0], 'fro') for h in history] + [np.nan] + [np.linalg.norm(X - h['Xs'][0], 'fro') for h in history2])
pgf.add('l_grad_norm', [h['lagrangian_grad'] for h in history] + [np.nan] + [h['lagrangian_grad'] for h in history2])
pgf.add('epsilon', [h['epsilon'] for h in history] + [np.nan] + [np.nan for h in history2])

pgf.write('data/fig_convergence.dat')



