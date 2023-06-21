import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File("NS_isotropic_90_90_90.h5")
ts = f["Turbulence/Ek"].keys()

ek = np.ndarray((len(ts), len(f["Turbulence/Ek/10"])))

for t in ts:
    ek[(int(t)-1), :] = np.array(f["Turbulence/Ek/{}".format(t)])

ts = np.arange(len(ts))

ks = np.arange(90)[:-14]

ek4 = ek[499][:-14]

# bins = f["Turbulence/bins"][:]


e_kol = (9/11)*3**(2/3)*(ks)**(-5/3)
print(np.log(ek4))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

ax1.plot(np.log(ks), np.log(ek4), label="measured")
ax1.plot(np.log(ks), np.log(e_kol), label=r"$E(k) = C\epsilon^{\frac{2}{3}}k^{-\frac{5}{3}}$")
ax1.set_ylabel(r"$\mathrm{log}(E(k))$")
ax1.axvline(np.log(3), label="$k_f$ = 3", color="C3")
ax1.axvline(np.log(70), label=r"$k_\nu$ = 70", color="C4")
ax2.plot(np.log(ks), np.log(ek4/e_kol))
ax2.set_ylabel(r"$\mathrm{log}(E(k)\times C^{-1}\epsilon^{-\frac{2}{3}}k^{\frac{5}{3}})$")
ax2.axvline(np.log(3), label="$k_f$ = 3", color="C3")
ax2.axvline(np.log(70), label=r"$k_\nu$ = 70", color="C4")
ax2.set_xlabel(r"$\mathrm{log}(kL)$")
fig.legend()
plt.savefig("spectrum1.pdf")
plt.show()
