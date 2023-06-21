import numpy as np
from spectralDNS import config, get_solver, solve
import matplotlib.pyplot as plt
import h5py
from math import ceil, sqrt

def initialize(UB_hat, UB, U, B, X, **context):
    params = config.params
    Um = 0.5*(params.U1 - params.U2)
    N = params.N
    U[1] = params.A*np.sin(params.k*X[0])
    U[2] = 0
    U[0, :, :N[1]//4] = params.U1 - Um*np.exp((X[1][:, :N[1]//4] - 0.5*np.pi)/params.delta)
    U[0, :, N[1]//4:N[1]//2] = params.U2 + Um*np.exp(-1.0*(X[1][:, N[1]//4:N[1]//2] - 0.5*np.pi)/params.delta)
    U[0, :, N[1]//2:3*N[1]//4] = params.U2 + Um*np.exp((X[1][:, N[1]//2:3*N[1]//4] - 1.5*np.pi)/params.delta)
    U[0, :, 3*N[1]//4:] = params.U1 - Um*np.exp(-1.0*(X[1][:, 3*N[1]//4:] - 1.5*np.pi)/params.delta)
    B[0, :, :, :] = params.B0
    B[1, :, :, :] = 0
    B[2, :, :, :] = params.B0



    UB_hat = UB.forward(UB_hat)

def update(context):
    pass


if __name__ == '__main__':
    config.update(
        {'nu': 0.000625,             # Viscosity
         'dt': 0.01,                 # Time step
         'T': 20.0,                   # End time
         'eta': 0.01,
         'M': [6, 6, 6],
         'k': 2,
         'A': 0.01,
         'delta': 0.1,
         'write_result': 1,
         'B0': 0.001,
         'U1': 1,
         'U2': -1,
         'convection': 'Divergence'})
    solver = get_solver(update=update)
    context = solver.get_context()
    initialize(**context)
    solve(solver, context)
