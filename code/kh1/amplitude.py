import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from math import sqrt
import argparse
from scipy.stats import linregress

parser = argparse.ArgumentParser(prog='plotter')
parser.add_argument('wavenumber')
parser.add_argument("-c", "--cutoff", default=0)
args = parser.parse_args()

k = int(args.wavenumber)

f = h5py.File("NS2D_amplitude.h5")

T = 0.5 * 2*np.pi
dt = 0.005

t0, tf = 0.0, np.pi
t0i = int(t0/dt)
tfi = int(tf/dt)

print(t0, tf)


amp = np.array(f["Amplitude/" + str(k)])[1]
print(amp.shape)

amp = amp[t0i:tfi]
x = np.linspace(t0, tf, len(amp))

lamp = np.log(amp)

d_dx = FinDiff(0, x[1]-x[0])

dlamp = d_dx(lamp)

# mask = np.logical_and(0.5*k < dlamp, dlamp < 2*k)
begin = np.argmax(dlamp > 0.5*k)
print(dlamp > 0.5*k)
end = np.argmax(np.logical_not(dlamp[begin:] > 0.5*k))
print(np.logical_not(dlamp[begin:] > 0.5*k))
# begin = 0
# end = 1000
print(begin, end)

a, b, r, _, _ = linregress(x[begin:end], lamp[begin:end])

# plt.plot(x, amp)
# plt.plot(x, a*np.exp(b*x))
# plt.plot(x[dlamp > dlamp[0]], lamp[dlamp > dlamp[0]])
plt.plot(x, lamp)
plt.plot(x[begin:end], dlamp[begin:end])
plt.plot(x, a*x + b)
print(a, b)
plt.show()

