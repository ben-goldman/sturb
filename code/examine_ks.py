import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.optimize import curve_fit
from math import sqrt
import argparse

T = 0.5
dt = 0.0001
n = T/dt


f = h5py.File("NS2D_amplitude.h5")
Nk = len(f["Amplitude"].keys())

def compute(k, T, f):
    amp = np.array(f["Amplitude/" + str(k)])[1]
    x = np.linspace(0, T, len(amp))
    dx = x[1] - x[0]
    print(T, dx)
    d_dx = FinDiff(0, dx, 1)
    da_dx = d_dx(amp)
    sigma = da_dx/amp
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    return sigma_mean, sigma_std

Nk = 10
ks = np.arange(1, Nk + 1)
sigmas = np.ndarray((2, len(ks)))
sigma_std = np.ndarray(len(ks))

for k in ks:
    sigmas[:, k-1] = compute(k, T, f)

a, b = np.polyfit(ks, sigmas[0], 1)
print(a, b)

plt.scatter(ks, sigmas[0], label="Mean growth rate")
plt.scatter(ks, sigmas[1], label="Stdev of growth rate")
plt.plot(ks, a*ks + b, label="Linear fit")
plt.legend()
# plt.savefig("KH1.eps")
plt.show()



