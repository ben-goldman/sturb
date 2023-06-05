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
    U[0, :, :N[1]//2] = np.tanh((X[1][:, :N[1]//2] - 0.5*np.pi)/params.delta)
    U[0, :, N[1]//2:] = -np.tanh((X[1][:, N[1]//2:] - 1.5*np.pi)/params.delta)
    for i in range(2):
        U_hat[i] = U[i].forward(U_hat[i])

    # U_hat[:] -= (K[0]*U_hat[0] + K[1]*U_hat[1])*K_over_K2
    # if solver.rank == 0:
        # U_hat[:, 0, 0] = 0.0
    # T.mask_nyquist(U_hat, mask)


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
    if params.tstep % params.plot_result == 0 and params.plot_result > 0:
        P = solver.get_pressure(**context)
        curl = solver.get_curl(**context)
    if params.tstep % params.compute_energy == 0:
        U = solver.get_velocity(**context)
        div_u = solver.get_divergence(**context)
        du = L2_norm(div_u, solver.comm)
        kk = solver.comm.reduce(np.sum(U.astype(np.float64)
                                       * U.astype(np.float64))
                                * dx[0]*dx[1]/L[0]/L[1]/2)
        if solver.rank == 0:
            print(params.tstep, kk)

    if params.tstep == 1 and params.plot_result > 0:
        fig, ax = plt.subplots(1, 1)
        fig.suptitle("Pressure")
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        im = ax.imshow(np.zeros((N[0], N[1])), cmap=plt.cm.bwr,
                       extent=[0, L[0], 0, L[1]])
        plt.colorbar(im)
        plt.draw()

        fig2, ax2 = plt.subplots(1, 1)
        fig2.suptitle("Vorticity")
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        im2 = ax2.imshow(np.zeros((N[0], N[1])), cmap=plt.cm.bwr,
                         extent=[0, L[0], 0, L[1]])
        plt.colorbar(im2)
        plt.draw()
        globals().update(dict(im=im, im2=im2))

    if params.tstep % params.plot_result == 0 and params.plot_result > 0:
        count += 1
        im.set_data(P[:, :].T)
        im.autoscale()
        plt.pause(0e-6)
        im2.set_data(curl[:, :].T)
        im2.autoscale()
        plt.pause(1e-6)
        plt.savefig("KH_{}.png".format(count))
        if solver.rank == 0:
            print(params.tstep)

    U = context.U[1, 0, 16]
    amp[..., params.tstep - 1] = U


if __name__ == "__main__":
    config.update(
            {
                'nu': 1.0e-5,
                'dt': 0.01,
                'T': 2.0,
                'A': 0.002,
                'delta': 0.01,
                'write_result': 1,
                'plot_result': -1,
                'compute_energy': 100,
                'N': [64, 64],
                'optimizer': 'cython',
                'amplitude_name': 'NS2D_amplitude.h5',
                'k': 1
                }, 'doublyperiodic'
            )
    solver = get_solver(update=update,
                        mesh='doublyperiodic',
                        parse_args=['NS2D'])
    context = solver.get_context()
    initialize(**context)
    amp = np.ndarray((ceil(config.params.T/config.params.dt)))
    solve(solver, context)
    f = h5py.File(config.params.amplitude_name, mode="a",
                  driver="mpio", comm=solver.comm)
    try:
        f.create_group("Amplitude")
    except ValueError:
        pass
    f["Amplitude"].create_dataset(str(config.params.k), data=amp)
    f.close()
