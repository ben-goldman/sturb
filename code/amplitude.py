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

T = 0.5
dt = 0.0001
n = T/dt

amp = np.array(f["Amplitude/" + str(k)])[1]

print(amp[0])

if cutoff:
    frac = cutoff/n
    T *= frac
    amp = amp[:cutoff]

x = np.linspace(0, T, len(amp))

dx = x[1] - x[0]

d_dx = FinDiff(0, dx, 2)

da_dx = d_dx(amp)
sigma2 = da_dx/amp
sigma = np.sqrt(sigma2)

(a, b), _ = curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  amp, p0 = [0.001, 3])
print(a, b)

# plt.plot(x, amp)
# plt.plot(x, a*np.exp(b*x))
plt.plot(sigma)
plt.show()
print(np.mean(sigma2))

