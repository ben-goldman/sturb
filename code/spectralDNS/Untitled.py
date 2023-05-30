#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from numpy import zeros, pi, sum, exp, sin, float64, prod
from spectralDNS import config, get_solver, solve


# In[2]:


def initialize(X, U, U_hat, mask, T, K, K_over_K2, **context):
    params = config.params
    Um = 0.5*(params.U1 - params.U2)
    N = params.N
    U[1] = params.A*sin(2*X[0])
    U[0, :, :N[1]//4] = params.U1 - Um*exp((X[1][:, :N[1]//4] - 0.5*pi)/params.delta)
    U[0, :, N[1]//4:N[1]//2] = params.U2 + Um*exp(-1.0*(X[1][:, N[1]//4:N[1]//2] - 0.5*pi)/params.delta)
    U[0, :, N[1]//2:3*N[1]//4] = params.U2 + Um*exp((X[1][:, N[1]//2:3*N[1]//4] - 1.5*pi)/params.delta)
    U[0, :, 3*N[1]//4:] = params.U1 - Um*exp(-1.0*(X[1][:, 3*N[1]//4:] - 1.5*pi)/params.delta)

    for i in range(2):
        U_hat[i] = U[i].forward(U_hat[i])

    U_hat[:] -= (K[0]*U_hat[0]+K[1]*U_hat[1])*K_over_K2
    if solver.rank == 0:
        U_hat[:, 0, 0] = 0.0

    T.mask_nyquist(U_hat, mask)

def L2_norm(u, comm):
    r"""Compute the L2-norm of real array a

    Computing \int abs(u)**2 dx

    """
    N = config.params.N
    result = comm.allreduce(sum(u**2))
    return result/prod(N)

def update(context):
    params = config.params
    solver = config.solver
    dx, L, N = params.dx, params.L, params.N

 


# In[4]:


config.update(
    {'nu': 1.0e-05,
     'dt': 0.007,
     'T': 25.0,
     'U1':-0.5,
     'U2':0.5,
     'l0': 0.001,    # Smoothing parameter
     'A': 0.01,      # Amplitude of perturbation
     'delta': 0.1,   # Width of perturbations
     'write_result': 500,
     'compute-energy': 50,
    }, 'doublyperiodic'
)
# Adding new arguments required here to allow overloading through commandline
# config.doublyperiodic.add_argument('--compute_energy', type=int, default=50)
solver = get_solver(update=update, mesh='doublyperiodic', parse_args=["NS2D"])
context = solver.get_context()
initialize(**context)
solve(solver, context)


# In[ ]:




