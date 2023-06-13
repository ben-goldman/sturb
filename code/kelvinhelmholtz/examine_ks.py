import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.stats import linregress
from math import sqrt
import argparse

plt.rcParams['text.usetex'] = True

f = h5py.File("NS2D_amplitude.h5")
print(f["Amplitude"].keys())
Nk = len(f["Amplitude"].keys())

def compute(k):
    print(k)
    f = h5py.File("NS2D_amplitude.h5")
    T = 0.5 * 2*np.pi
    dt = 0.005
    t0, tf = 0.0, np.pi
    t0i = int(t0/dt)
    tfi = int(tf/dt)
    amp = np.array(f["Amplitude/" + str(k)])[1]
    amp = amp[t0i:tfi]
    x = np.linspace(t0, tf, len(amp))
    lamp = np.log(amp)
    d_dx = FinDiff(0, x[1]-x[0])
    dlamp = d_dx(lamp)
    begin = np.argmax(dlamp > 0.5*k)
    end = np.argmax(np.logical_not(dlamp[begin:] > 0.5*k))
    print(begin, end)
    a, b, r, _, _ = linregress(x[begin:end], lamp[begin:end])
    print(a, b, r)
    return a

Nk = 8
ks = np.arange(1, Nk + 1)
sigmas = np.ndarray((len(ks)))

for k in ks:
    sigmas[k-1] = compute(k)

a, b, r, _, _= linregress(ks, sigmas)
print()
print(a, b)

plt.scatter(ks, sigmas, label="Mean growth rate")
plt.plot(ks, a*ks + b, label=r"Linear fit $y = {:n}k + {:n}$, $r={:n}$".format(a, b, r))
plt.legend()
plt.xlabel("Perturbation Wavenumber $k$")
plt.ylabel(r"Amplitude growth rate $\times \frac{L}{U}$")
plt.savefig("KH1.pdf")
plt.show()



