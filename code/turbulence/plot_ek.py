import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File("NS_isotropic_90_90_90.h5")
ts = f["Turbulence/Ek"].keys()

ek = np.ndarray((len(ts), len(f["Turbulence/Ek/10"])))

for t in ts:
    ek[(int(t)-1), :] = np.array(f["Turbulence/Ek/{}".format(t)])

ek = ek[400]


# bins = f["Turbulence/bins"][:]
ks = np.arange(45)

e_kol = 1.5*(ks/2)**(-5/3)

plt.plot(np.log(ks/70), np.log(ek))
plt.plot(np.log(ks/70), np.log(e_kol))
plt.show()
