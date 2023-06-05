import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.optimize import curve_fit
from math import sqrt
import argparse

f = h5py.File("NS2D_amplitude.h5")
print(f["Amplitude"].keys())

T = 1.0
dt = 0.001
n = T/dt


Nk = len(f["Amplitude"].keys())


def compute(k):
    amps = np.array(f["Amplitude/" + str(k)])[1]
    kx = np.abs(amps[:, 0]).argmax()
    amp = amps[kx]
    print(amp[0])
    x = np.linspace(0, T, len(amp))
    dx = x[1] - x[0]
    d2_dx2 = FinDiff(0, dx, 2)

    d2a_dx2 = d2_dx2(amp)

    sigma = d2a_dx2/amp

    sigma = np.sqrt(np.mean(sigma[:100]))
    return sigma


ks = np.arange(1, Nk + 1)
sigmas = np.ndarray(len(ks))

for k in ks:
    sigmas[k-1] = compute(k)

plt.scatter(ks, sigmas)
plt.show()
