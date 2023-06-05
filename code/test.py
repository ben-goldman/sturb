import numpy as np
import h5py
import matplotlib.pyplot as plt
from findiff import FinDiff
from scipy.optimize import curve_fit
from math import sqrt
import argparse

f = h5py.File("NS2D_w.h5")

U = np.array([f["U0/2D/200"][:, :], f["U1/2D/200"][:, :]])

U1 = U[1, :, :]

amp = np.std(U1[:, :])*sqrt(2)

plt.plot(U1[:, 0])
print(amp)
plt.show()
