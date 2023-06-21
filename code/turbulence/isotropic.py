"""
Homogeneous turbulence. See [1] for initialization and [2] for a section
on forcing the lowest wavenumbers to maintain a constant turbulent
kinetic energy.

[1] R. S. Rogallo, "Numerical experiments in homogeneous turbulence,"
NASA TM 81315 (1981)

[2] A. G. Lamorgese and D. A. Caughey and S. B. Pope, "Direct numerical simulation
of homogeneous turbulence with hyperviscosity", Physics of Fluids, 17, 1, 015106,
2005, (https://doi.org/10.1063/1.1833415)

"""
from __future__ import print_function
import warnings
import numpy as np
from numpy import pi, zeros, sum
from shenfun import Function
from shenfun.fourier import energy_fourier
from spectralDNS import config, get_solver, solve

try:
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None

def initialize(solver, context):
    c = context
    # Create mask with ones where |k| < Kf2 and zeros elsewhere
    kf = config.params.Kf2
    c.k2_mask = np.where(c.K2 <= kf**2, 1, 0)
    np.random.seed(solver.rank)
    k = np.sqrt(c.K2)
    k = np.where(k == 0, 1, k)
    kk = c.K2.copy()
    kk = np.where(kk == 0, 1, kk)
    k1, k2, k3 = c.K[0], c.K[1], c.K[2]
    ksq = np.sqrt(k1**2+k2**2)
    ksq = np.where(ksq == 0, 1, ksq)

    E0 = np.sqrt(9./11./kf*c.K2/kf**2)*c.k2_mask
    E1 = np.sqrt(9./11./kf*(k/kf)**(-5./3.))*(1-c.k2_mask)
    Ek = E0 + E1
    # theta1, theta2, phi, alpha and beta from [1]
    theta1, theta2, phi = np.random.sample(c.U_hat.shape)*2j*np.pi
    alpha = np.sqrt(Ek/4./np.pi/kk)*np.exp(1j*theta1)*np.cos(phi)
    beta = np.sqrt(Ek/4./np.pi/kk)*np.exp(1j*theta2)*np.sin(phi)
    c.U_hat[0] = (alpha*k*k2 + beta*k1*k3)/(k*ksq)
    c.U_hat[1] = (beta*k2*k3 - alpha*k*k1)/(k*ksq)
    c.U_hat[2] = beta*ksq/k
    c.mask = c.T.get_mask_nyquist()
    c.T.mask_nyquist(c.U_hat, c.mask)

    solver.get_velocity(**c)
    U_hat = solver.set_velocity(**c)

    K = c.K
    # project to zero divergence
    U_hat[:] -= (K[0]*U_hat[0]+K[1]*U_hat[1]+K[2]*U_hat[2])*c.K_over_K2

    if solver.rank == 0:
        c.U_hat[:, 0, 0, 0] = 0.0

    # Scale to get correct kinetic energy. Target from [2]
    energy = 0.5*energy_fourier(c.U_hat, c.T)
    target = config.params.Re_lam*(config.params.nu*config.params.kd)**2/np.sqrt(20./3.)
    c.U_hat *= np.sqrt(target/energy)
    energy = 0.5*energy_fourier(c.U_hat, c.T)

    if 'VV' in config.params.solver:
        c.W_hat = solver.cross2(c.W_hat, c.K, c.U_hat)

    config.params.t = 0.0
    config.params.tstep = 0
    c.target_energy = energy_fourier(c.U_hat, c.T)

def L2_norm(comm, u):
    r"""Compute the L2-norm of real array a

    Computing \int abs(u)**2 dx

    """
    N = config.params.N
    result = comm.allreduce(np.sum(u**2))
    return result/np.prod(N)

def spectrum(solver, context):
    c = context
    uiui = np.zeros(c.U_hat[0].shape)
    uiui[..., 1:-1] = 2*np.sum((c.U_hat[..., 1:-1]*np.conj(c.U_hat[..., 1:-1])).real, axis=0)
    uiui[..., 0] = np.sum((c.U_hat[..., 0]*np.conj(c.U_hat[..., 0])).real, axis=0)
    uiui[..., -1] = np.sum((c.U_hat[..., -1]*np.conj(c.U_hat[..., -1])).real, axis=0)
    uiui *= (4./3.*np.pi)

    # Create bins for Ek
    Nb = 90 # int(np.sqrt(sum((config.params.N/2)**2)/3))
    bins = np.array(range(0, Nb))+0.5
    z = np.digitize(np.sqrt(context.K2), bins, right=True)

    # Sample
    Ek = np.zeros(Nb)
    ll = np.zeros(Nb)
    for i, k in enumerate(bins[1:]):
        k0 = bins[i] # lower limit, k is upper
        ii = np.where((z > k0) & (z <= k))
        ll[i] = len(ii[0])
        Ek[i] = (k**3 - k0**3)*np.sum(uiui[ii])

    Ek = solver.comm.allreduce(Ek)
    ll = solver.comm.allreduce(ll)
    for i in range(Nb):
        if not ll[i] == 0:
            Ek[i] = Ek[i] / ll[i]

    E0 = uiui.mean(axis=(1, 2))
    E1 = uiui.mean(axis=(0, 2))
    E2 = uiui.mean(axis=(0, 1))

    ## Rij
    #for i in range(3):
    #    c.U[i] = c.FFT.ifftn(c.U_hat[i], c.U[i])
    #X = c.FFT.get_local_mesh()
    #R = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    ## Sample
    #Rii = np.zeros_like(c.U)
    #Rii[0] = c.FFT.ifftn(np.conj(c.U_hat[0])*c.U_hat[0], Rii[0])
    #Rii[1] = c.FFT.ifftn(np.conj(c.U_hat[1])*c.U_hat[1], Rii[1])
    #Rii[2] = c.FFT.ifftn(np.conj(c.U_hat[2])*c.U_hat[2], Rii[2])

    #R11 = np.sum(Rii[:, :, 0, 0] + Rii[:, 0, :, 0] + Rii[:, 0, 0, :], axis=0)/3

    #Nr = 20
    #rbins = np.linspace(0, 2*np.pi, Nr)
    #rz = np.digitize(R, rbins, right=True)
    #RR = np.zeros(Nr)
    #for i in range(Nr):
    #    ii = np.where(rz == i)
    #    RR[i] = np.sum(Rii[0][ii] + Rii[1][ii] + Rii[2][ii]) / len(ii[0])

    #Rxx = np.zeros((3, config.params.N[0]))
    #for i in range(config.params.N[0]):
    #    Rxx[0, i] = (c.U[0] * np.roll(c.U[0], -i, axis=0)).mean()
    #    Rxx[1, i] = (c.U[0] * np.roll(c.U[0], -i, axis=1)).mean()
    #    Rxx[2, i] = (c.U[0] * np.roll(c.U[0], -i, axis=2)).mean()

    return Ek, bins, E0, E1, E2

k = []
w = []
kold = zeros(1)
im1 = None
energy_new = None
def update(context):
    global k, w, im1, energy_new
    c = context
    params = config.params
    solver = config.solver

    energy_new = energy_fourier(c.U_hat, c.T)
    energy_lower = energy_fourier(c.U_hat*c.k2_mask, c.T)
    energy_upper = energy_new - energy_lower

    print(params.tstep, energy_new, energy_lower, energy_upper, c.target_energy)
    alpha2 = (c.target_energy - energy_upper) /energy_lower
    alpha = np.sqrt(alpha2)

    #du = c.U_hat*c.k2_mask*(alpha)
    #dus = energy_fourier(du*c.U_hat, c.T)

    energy_old = energy_new

    #c.dU[:] = alpha*c.k2_mask*c.U_hat
    c.U_hat *= (alpha*c.k2_mask + (1-c.k2_mask))

    energy_new = energy_fourier(c.U_hat, c.T)

    assert np.sqrt((energy_new-c.target_energy)**2) < 1e-7, np.sqrt((energy_new-c.target_energy)**2)

    if params.tstep % params.compute_spectrum == 0:
        Ek, _, _, _, _ = spectrum(solver, context)
        f = h5py.File(context.spectrumname, driver='mpio', comm=solver.comm, mode='a')
        f['Turbulence/Ek'].create_dataset(str(params.tstep), data=Ek)
        f.close()

def init_from_file(filename, solver, context):
    f = h5py.File(filename, driver="mpio", comm=solver.comm)
    assert "0" in f["U/3D"]
    U_hat = context.U_hat
    s = context.T.local_slice(True)

    U_hat[:] = f["U/3D/0"][:, s[0], s[1], s[2]]
    if solver.rank == 0:
        U_hat[:, 0, 0, 0] = 0.0

    if 'VV' in config.params.solver:
        context.W_hat = solver.cross2(context.W_hat, context.K, context.U_hat)

    context.target_energy = energy_fourier(U_hat, context.T)

    f.close()


if __name__ == "__main__":
    import h5py
    small = [64, 64, 64]
    big = [90, 90, 90]
    config.update(
        {'dt': 0.002,                 # Time step
         'T': 1,                      # End time
         'L': [2.*pi, 2.*pi, 2.*pi],
         'checkpoint': 100,
         'write_result': 1e8,
         'dealias': '3/2-rule',
         'solver': 'NS',
         'compute_spectrum': 1,
         'plot_step': 1000,
         'N': big,
         'Re_lam': 105,
         'Kf2': 3,
         'kd': 70
        }, "triplyperiodic"
    )
    # config.triplyperiodic.add_argument("--N", default=[60, 60, 60], nargs=3,
    #                                    help="Mesh size. Trumps M.")
    # config.triplyperiodic.add_argument("--compute_energy", type=int, default=100)
    # config.triplyperiodic.add_argument("--compute_spectrum", type=int, default=1000)
    # config.triplyperiodic.add_argument("--plot_step", type=int, default=1000)
    # config.triplyperiodic.add_argument("--Kf2", type=int, default=3)
    # config.triplyperiodic.add_argument("--kd", type=float, default=50.)
    # config.triplyperiodic.add_argument("--Re_lam", type=float, default=84.)
    sol = get_solver(update=update, mesh="triplyperiodic")
    config.params.nu = (1./config.params.kd**(4./3.))

    context = sol.get_context()
    initialize(sol, context)
    #init_from_file("NS_isotropic_60_60_60_c.h5", sol, context)
    context.hdf5file.filename = "NS_isotropic_{}_{}_{}".format(*config.params.N)

    Ek, bins, E0, E1, E2 = spectrum(sol, context)
    context.spectrumname = context.hdf5file.filename+".h5"
    f = h5py.File(context.spectrumname, mode='w', driver='mpio', comm=sol.comm)
    f.create_group("Turbulence")
    f["Turbulence"].create_group("Ek")
    bins = np.array(bins)
    f["Turbulence"].create_dataset("bins", data=bins)
    f.close()
    solve(sol, context)
