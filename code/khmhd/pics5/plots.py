import numpy as np, matplotlib.pyplot as plt, h5py
f = h5py.File("../MHD_w.h5")
for t in range(1, 50):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(5, 10))
    t = str(t*100)
    u2 = np.array(f[f"UB0/3D/{t}"])**2 + np.array(f[f"UB1/3D/{t}"])**2 + np.array(f[f"UB2/3D/{t}"])**2
    b2 = np.array(f[f"UB3/3D/{t}"])**2 + np.array(f[f"UB4/3D/{t}"])**2 + np.array(f[f"UB5/3D/{t}"])**2
    c1 = ax1.imshow(np.log(u2[:, :, 64]))
    c2 = ax2.imshow(np.log(b2[:, :, 64]))
    fig.colorbar(c1)
    fig.colorbar(c1)
    ax1.set_title(r"$\vec{u}^2$")
    ax2.set_title(r"$\vec{b}^2$")
    fig.suptitle(f"$t={t}$")
    fname = f"b2_{int(t):04d}.png"
    plt.savefig(fname)
    plt.close()
    print(fname)
