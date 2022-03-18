import numpy as np
import matplotlib.pyplot as plt

x1 = np.load('Video_Tracking_data/Trial1/DPmean_data_RB0.npy') #mean value of upper link angle (deg)
x2 = np.load('Video_Tracking_data/Trial1/DPmean_data_RB1.npy') #mean value of upper link angle (deg)

t = x1[0]
x1 = x1[1] * (np.pi/180) # Convert to radians
x2 = x2[1] * (np.pi/180) # "
fig, axes = plt.subplots(2)
axes[0].plot(t, x1)
axes[1].plot(t, x2)
fig, axes = plt.subplots()
axes.plot(x1, x2)
plt.show()


