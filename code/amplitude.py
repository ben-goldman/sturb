import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.optimize import curve_fit
from math import sqrt
import argparse

parser = argparse.ArgumentParser(prog='plotter')
parser.add_argument('wavenumber')
parser.add_argument("-c", "--cutoff", default=0)

fit = False

args = parser.parse_args()

k = int(args.wavenumber)
cutoff = int(args.cutoff)

f = h5py.File("NS2D_amplitude.h5")

T = 0.1
dt = 0.001
n = T/dt

amp = np.array(f["Amplitude/" + str(k)])[1]
amp0 = np.array(f["Amplitude/" + str(k)])[0]
print(amp.shape)
print(amp[0])

if cutoff:
    frac = cutoff/n
    T *= frac
    amp = amp[:cutoff]

x = np.linspace(0, T, len(amp))
dx = x[1] - x[0]

if fit:
    guess = [0.0001, 1, 0.01]
    popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, amp,
                           p0=guess)

    a = popt[0]
    b = popt[1]
    c = popt[2]
    print(a, b, c)
    amp_fit = a * np.exp(b * x) + c
else:
    amp_fit = amp

print(n, dx)
d2_dx2 = FinDiff(0, dx, 2)

d2a_dx2 = d2_dx2(amp_fit)

sigma = d2a_dx2/amp_fit

plt.plot(amp)
plt.plot(amp0)
# plt.plot(amp_fit)
# plt.plot(sigma[abs(sigma) < 1e1])
plt.show()

sigma = np.mean(sigma[abs(sigma) < 1e1])
print(k, sigma, sqrt(sigma))
