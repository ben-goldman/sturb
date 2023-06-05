import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from math import sqrt
import argparse
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(prog='plotter')
parser.add_argument('wavenumber')
parser.add_argument("-c", "--cutoff", default=0)

fit = False

args = parser.parse_args()

k = int(args.wavenumber)
cutoff = int(args.cutoff)

f = h5py.File("NS2D_amplitude.h5")

T = 1.0
dt = 0.001
n = T/dt

amps = np.array(f["Amplitude/" + str(k)])[1]
amps0 = np.array(f["Amplitude/" + str(k)])[0]
print(amps[:, 0])

kx = np.abs(amps[:, 0]).argmax()
print(kx)
amp = amps[kx]

print(amp[0])

# amp = np.sum(amps, axis=0)

if cutoff:
    frac = cutoff/n
    T *= frac
    amp = amp[:cutoff]

x = np.linspace(0, T, len(amp))

dx = x[1] - x[0]

d2_dx2 = FinDiff(0, dx, 2)

d2a_dx2 = d2_dx2(amp)

sigma = d2a_dx2/amp

# for i in range(8):
    # plt.plot(amps[i])
    # plt.plot(amps0[i])
plt.plot(amp)
# plt.plot(sigma)
plt.show()

sigma = np.mean(sigma[abs(sigma) < 1e1])
print(k, sigma, sqrt(abs(sigma)))


# (a, b, c), pcov = curve_fit(lambda t, a, b, c: a*np.exp(b*t) + c, x, amp, p0 = (0.01, 2, 0.1))

# print(a, b, c)
# plt.plot(a*np.exp(b*x) + c)
