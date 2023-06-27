import numpy as np
from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
import h5py
from math import ceil, sqrt

def initialize(UB_hat, UB, U, B, X, **context):
    params = config.params
    N = params.N
    print(N)
    k = 2*np.pi*np.array([0, np.cos(params.theta_p), np.sin(params.theta_p)])
    vx = params.A*np.cos(np.tensordot(k, X, axes=1) + np.random.rand())
    vy = params.A*np.cos(np.tensordot(k, X, axes=1) + np.random.rand())
    Um = 0.5*(params.U1 - params.U2)
    U[0, :, :N[1]//4] = params.U1 - Um*np.exp((X[1][:, :N[1]//4] - 0.5*np.pi)/params.delta)
    U[0, :, N[1]//4:N[1]//2] = params.U2 + Um*np.exp(-1.0*(X[1][:, N[1]//4:N[1]//2] - 0.5*np.pi)/params.delta)
    U[0, :, N[1]//2:3*N[1]//4] = params.U2 + Um*np.exp((X[1][:, N[1]//2:3*N[1]//4] - 1.5*np.pi)/params.delta)
    U[0, :, 3*N[1]//4:] = params.U1 - Um*np.exp(-1.0*(X[1][:, 3*N[1]//4:] - 1.5*np.pi)/params.delta)
    U[1] = vx
    U[2] = vy
    B[0, :, :, :] = params.B0
    B[1, :, :, :] = 0
    B[2, :, :, :] = params.B0



    UB_hat = UB.forward(UB_hat)

def update(context):
    params = config.params
    solver = config.solver
    dx, L, N = params.dx, params.L, params.N
    UB = context.UB_hat.backward(context.UB)
    B = UB[3:]
    B2 = solver.comm.allreduce(np.mean(B[0]**2 + B[1]**2 + B[2]**2))
    amp[params.tstep - 1] = B2
    print(f"{str(params.tstep)} : {str(len(amp))}")


if __name__ == '__main__':
    config.update(
        {'nu': 0.0005,             # Viscosity
         'dt': 0.01,                 # Time step
         'T': 50.0,                   # End time
         'eta': 0.0001,
         'M': [8, 8, 8],
         'L': [2*np.pi, 2*np.pi, 2*np.pi],
         'A': 0.01,
         'delta': 0.1,
         'write_result': 10,
         'B0': 0.001,
         'U1': 1,
         'U2': -1,
         'theta_p': 0.005,
         'solver': "MHD",
         'amplitude_name': "../../../out/dynamo.h5",
         'optimization': 'cython',
         'convection': 'Divergence'})

    solver = get_solver(update=update)
    context = solver.get_context()
    context.hdf5file.filename = "../../../out/MHD_1"
    initialize(**context)
    amp = np.ndarray((ceil(config.params.T/config.params.dt) + 1))
    solve(solver, context)
    f = h5py.File(config.params.amplitude_name, mode="a",
                  driver="mpio", comm=solver.comm)
    f.create_dataset("B2", data=amp)
    f.close()
