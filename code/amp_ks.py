import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.optimize import curve_fit
from math import sqrt
import argparse

f = h5py.File("NS2D_amplitude.h5")

T = 0.1
dt = 0.0001
n = T/dt

Nk = 10


def compute(k):
    amp = np.array(f["Amplitude/" + str(k)])[1]
    x = np.linspace(0, T, len(amp))
    dx = x[1] - x[0]
    d2_dx2 = FinDiff(0, dx, 2)

    d2a_dx2 = d2_dx2(amp)

    sigma = d2a_dx2/amp

    sigma = np.mean(sigma[:100])
    return sigma


ks = np.arange(1, Nk)
sigmas = np.ndarray(len(ks))

for k in ks:
    sigmas[k-1] = compute(k)

plt.scatter(ks, sigmas)
plt.show()
