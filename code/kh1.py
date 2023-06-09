import numpy as np
from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
import h5py
from math import ceil, sqrt



def initialize(X, U, U_hat, **context):
    params = config.params
    N = params.N
    U[1] = params.A*np.sin(params.k*X[0])*np.exp(-((X[1] - np.pi)*params.k)**2)
    U[0, :, :] = np.tanh((X[1][:, :] - np.pi)/params.delta)

    U_hat = U.forward(U_hat)


def update(context):
    global im, im2, count
    params = config.params
    solver = config.solver
    dx, L, N = params.dx, params.L, params.N
    u = solver.get_velocity(**context)[:, :, N[1]//2].copy()
    u_std = solver.comm.allreduce(np.std(u, axis=1)) * np.sqrt(2)
    amp[..., params.tstep - 1] = u_std


if __name__ == "__main__":
    config.update(
            {
                'nu': 1.0e-20,
                'dt': 0.005,
                'T': 0.5*2*np.pi,
                'A': 0.01,
                'delta': 0.01,
                'write_result': 100,
                'plot_result': -1,
                'compute_energy': 100,
                'N': [512, 512],
                'optimizer': 'cython',
                'amplitude_name': 'NS2D_amplitude.h5',
                'solver': 'NS2D'
                }, 'doublyperiodic'
            )
    config.doublyperiodic.add_argument('--k', type=int, default=1)
    solver = get_solver(update=update,
                        mesh='doublyperiodic')
    context = solver.get_context()
    initialize(**context)
    amp = np.ndarray((2, ceil(config.params.T/config.params.dt)))
    solve(solver, context)
    f = h5py.File(config.params.amplitude_name, mode="a",
                  driver="mpio", comm=solver.comm)
    try:
        f.create_group("Amplitude")
    except ValueError:
        pass
    f["Amplitude"].create_dataset(str(config.params.k), data=amp)
    f.close()
