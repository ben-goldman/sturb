import h5py
import numpy as np
import matplotlib.pyplot as plt
f = h5py.File("../MHD_w.h5")
for y in range(64):
        t = '1000'
        U2 = f["UB0/3D/"+t][:, :, :]**2 + f["UB1/3D/"+t][:, :, :]**2 + f["UB2/3D/"+t][:, :, :]**2
        B2 = f["UB3/3D/"+t][:, :, :]**2 + f["UB4/3D/"+t][:, :, :]**2 + f["UB5/3D/"+t][:, :, :]**2
        c = plt.imshow(B2[:, :, y])
        plt.colorbar(c)
        plt.suptitle("B^2/"+t)
        plt.savefig(f"By{int(y):04d}")
        print(f"By{int(y):04d}")
        plt.close()
        U2 = np.mean(U2)
        B2 = np.mean(B2)
