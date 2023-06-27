import numpy as np, matplotlib.pyplot as plt, h5py, matplotlib.colors as colors, matplotlib as mpl
f = h5py.File("../MHD_w.h5")
ts = f["UB1/3D"].keys()
tmin = min(ts)
tmax = max(ts)
u2min = 0
b2min = (np.array(f[f"UB3/3D/{tmin}"])**2 + np.array(f[f"UB4/3D/{tmin}"])**2 + np.array(f[f"UB5/3D/{tmin}"])**2).min()
u2max = 1
b2max = (np.array(f[f"UB3/3D/{tmax}"])**2 + np.array(f[f"UB4/3D/{tmax}"])**2 + np.array(f[f"UB5/3D/{tmax}"])**2).max()
for t in ts:
    if int(t) % 10 == 0:
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 6))
        u2 = np.array(f[f"UB0/3D/{t}"])**2 + np.array(f[f"UB1/3D/{t}"])**2 + np.array(f[f"UB2/3D/{t}"])**2
        b2 = np.array(f[f"UB3/3D/{t}"])**2 + np.array(f[f"UB4/3D/{t}"])**2 + np.array(f[f"UB5/3D/{t}"])**2
        c1 = ax1.imshow(u2[:, :, 16], norm=colors.Normalize(vmin=u2min, vmax=u2max))
        c2 = ax2.imshow(b2[:, :, 16], norm=colors.LogNorm(vmin=b2min, vmax=b2max))
        print(b2.min(), b2.max())
        fig.colorbar(c1, ax=ax1, location="bottom")
        fig.colorbar(c2, ax=ax2, location="bottom")
        ax1.set_title(r"$\vec{u}^2$")
        ax2.set_title(r"$\vec{B}^2$")
        fig.suptitle(f"$t={t}$")
        fname = f"u2b2_{int(t):04d}.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(fname)
