import numpy as np, matplotlib.pyplot as plt, h5py
f = h5py.File("../MHD_w.h5")
for t in range(1, 50):
    t = str(t*100)
    B2 = np.array(f[f"UB3/3D/{t}"])**2 + np.array(f[f"UB4/3D/{t}"])**2 + np.array(f[f"UB5/3D/{t}"])**2
    c = plt.imshow(np.log(B2[:, :, 64]))
    plt.colorbar(c)
    plt.suptitle(f"$t={t}$")
    fname = f"b2_{int(t):04d}.png"
    plt.savefig(fname)
    plt.close()
    print(fname)

for t in range(1, 50):
    t = str(t*100)
    u2 = np.array(f[f"UB0/3D/{t}"])**2 + np.array(f[f"UB1/3D/{t}"])**2 + np.array(f[f"UB2/3D/{t}"])**2
    c = plt.imshow(np.log(u2[:, :, 64]))
    plt.colorbar(c)
    plt.suptitle(f"$t={t}$")
    fname = f"u2_{int(t):04d}.png"
    plt.savefig(fname)
    plt.close()
    print(fname)
