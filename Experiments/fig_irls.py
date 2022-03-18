import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

epsilon = 1e-3
q = 0
xhat = 0.1

x = np.linspace(0,1, 1000)

y = x**2*(x**2 + epsilon)**(q/2. - 1)

y_irls = x**2*(xhat**2 + epsilon)**(q/2.-1)

ax.plot(x,y)
ax.plot(x,y_irls)
plt.show()
