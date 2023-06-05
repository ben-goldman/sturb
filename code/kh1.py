import numpy as np
from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
import h5py
from math import ceil, sqrt


def rms(X):
    return np.sqrt(np.mean(X**2))


def initialize(X, U, U_hat, mask, T, K, K_over_K2, **context):
    params = config.params
    N = params.N
    U[1] = params.A*np.sin(params.k*X[0])
    U[0, :, :N[1]//2] = np.tanh((X[1][:, :N[1]//2]-0.5*np.pi)/params.delta)
    U[0, :, N[1]//2:] = -np.tanh((X[1][:, N[1]//2:]-1.5*np.pi)/params.delta)

    U_hat = U.forward(U_hat)

    U_hat[:] -= (K[0]*U_hat[0]+K[1]*U_hat[1])*K_over_K2
    if solver.rank == 0:
        U_hat[:, 0, 0] = 0.0

    T.mask_nyquist(U_hat, mask)


def L2_norm(u, comm):
    N = config.params.N
    result = comm.allreduce(sum(u**2))
    return result/np.prod(N)


im, im2 = None, None
count = 0


def update(context):
    global im, im2, count
    params = config.params
    solver = config.solver
    dx, L, N = params.dx, params.L, params.N
    if params.tstep % params.compute_energy == 0:
        U = solver.get_velocity(**context)
        # div_u = solver.get_divergence(**context)
        # du = L2_norm(div_u, solver.comm)
        kk = solver.comm.reduce(np.sum(U.astype(np.float64)
                                       * U.astype(np.float64))
                                * dx[0]*dx[1]/L[0]/L[1]/2)
        if solver.rank == 0:
            print(params.tstep, kk)

    u0 = solver.get_velocity(**context)[0, :, 32]
    u1 = solver.get_velocity(**context)[1, :, 32]
    amp[..., params.tstep - 1] = [u0, u1]


if __name__ == "__main__":
    config.update(
            {
                'nu': 1.0e-20,
                'dt': 0.001,
                'T': 1.0,
                'A': 0.01,
                'delta': 0.01,
                'write_result': 100,
                'plot_result': -1,
                'compute_energy': 100,
                'N': [128, 128],
                'optimizer': 'cython',
                'amplitude_name': 'NS2D_amplitude.h5',
                }, 'doublyperiodic'
            )
    config.doublyperiodic.add_argument('--k', type=int, default=1)
    solver = get_solver(update=update,
                        mesh='doublyperiodic')
    context = solver.get_context()
    initialize(**context)
    amp = np.ndarray((2, 8, ceil(config.params.T/config.params.dt)))
    solve(solver, context)
    f = h5py.File(config.params.amplitude_name, mode="a",
                  driver="mpio", comm=solver.comm)
    try:
        f.create_group("Amplitude")
    except ValueError:
        pass
    f["Amplitude"].create_dataset(str(config.params.k), data=amp)
    f.close()
