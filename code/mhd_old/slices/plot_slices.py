import h5py
import numpy as np
import matplotlib.pyplot as plt
f = h5py.File("../MHD_w.h5")
ts = f["UB1/3D"].keys()
for t in ts:
    if int(t) % 10 == 0:
        U2 = f["UB0/3D/"+t][:, :, :]**2 + f["UB1/3D/"+t][:, :, :]**2 + f["UB2/3D/"+t][:, :, :]**2
        B2 = f["UB3/3D/"+t][:, :, :]**2 + f["UB4/3D/"+t][:, :, :]**2 + f["UB5/3D/"+t][:, :, :]**2
        # plt.imshow(U2[:, :, 8])
        # plt.suptitle("U^2/"+t)
        # plt.savefig("U"+t+".png")
        # plt.close()
        c = plt.imshow(B2[:, :, 8])
        plt.colorbar(c)
        plt.suptitle("B^2/"+t)
        plt.savefig(f"B{int(t):04d}")
        print(f"B{int(t):04d}")
        plt.close()
        U2 = np.mean(U2)
        B2 = np.mean(B2)
