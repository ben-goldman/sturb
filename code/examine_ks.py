import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.optimize import curve_fit
from math import sqrt
import argparse


f = h5py.File("NS2D_amplitude.h5")
print(f["Amplitude"].keys())
Nk = len(f["Amplitude"].keys())

def compute(k, f):
    f = h5py.File("NS2D_amplitude.h5")
    T = 0.5 * 2*np.pi
    dt = 0.005
    t0, tf = 0.25, np.pi
    t0i = int(t0/dt)
    tfi = int(tf/dt)
    amp = np.array(f["Amplitude/" + str(k)])[1]
    amp = amp[t0i:tfi]
    x = np.linspace(t0, tf, len(amp))
    lamp = np.log(amp)
    d_dx = FinDiff(0, x[1]-x[0])
    dlamp = d_dx(lamp)
    print(k)
    print(dlamp[dlamp > dlamp[0]])
    a, b = np.polyfit(x[dlamp > dlamp[0]], lamp[dlamp > dlamp[0]], 1)
    print(k, a, b)
    return a

ks = np.arange(1, Nk + 1)
sigmas = np.ndarray((len(ks)))

for k in ks:
    sigmas[k-1] = compute(k, f)

a, b = np.polyfit(ks, sigmas, 1)
print()
print(a, b)

plt.scatter(ks, sigmas, label="Mean growth rate")
plt.plot(ks, a*ks + b, label="Linear fit")
plt.legend()
# plt.savefig("KH1.eps")
plt.show()



