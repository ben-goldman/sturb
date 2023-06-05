import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.optimize import curve_fit
from math import sqrt
import argparse

parser = argparse.ArgumentParser(prog='plotter')
parser.add_argument('wavenumber')

args = parser.parse_args()

k = int(args.wavenumber)

f = h5py.File("NS2D_amplitude.h5")

T = 2
dt = 0.01
n = T/dt

amp = np.array(f["Amplitude/" + str(k)])
print(amp.shape)


x = np.linspace(0, T, len(amp))
dx = x[1] - x[0]

a_guess = 0.001
b_guess = 1
popt, pcov = curve_fit(lambda t, a, b,: a * np.exp(b * t), x, amp, p0=(a_guess, b_guess))

a = popt[0]
b = popt[1]
print(a, b)
amp_fit = a * np.exp(b * x)

print(n, dx)
d2_dx2 = FinDiff(0, dx, 2)

d2a_dx2 = d2_dx2(amp_fit)

sigma = d2a_dx2/amp_fit

plt.plot(amp[:])
# plt.plot(d2a_dx2)
plt.show()

sigma = np.mean(sigma)
print(k, sqrt(sigma))
