import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from rodeo.ibm_init import ibm_init
from rodeo.ode_solve import *

def fitz(X_t, t, theta):
    "Fitz ODE written for jax"
    a, b, c = theta
    V, R = X_t[:, 0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                      [-1/c*(V - a + b*R)]])

def fitz0(X_t, t, theta):
    "Fitz ODE written for odeint"
    a, b, c = theta
    V, R = X_t
    return jnp.array([c*(V - V*V*V/3 + R),
                      -1/c*(V - a + b*R)])

def fitzsolve_example():
    # Produce a Pseudo-RNG key
    key = jax.random.PRNGKey(0)

    W = jnp.array([[[0., 1., 0.]], [[0., 1., 0.]]])  # LHS vector of ODE
    x0 = jnp.array([[-1., 1., 0.], [1., 1/3, 0.]])  # initial value for the IVP
    theta = jnp.array([.2, .2, 3]) # parameters of ODE

    # Time interval on which a solution is sought.
    tmin = 0.
    tmax = 40.

    # 3.  Define the prior process
    n_obs = 2
    n_deriv_prior = 3  # n_order = p from the paper
    n_order = jnp.array([n_deriv_prior]*n_obs)

    # IBM process scale factor
    sigma = jnp.array([.01]*n_obs)

    # 4.  Instantiate the ODE solver object.

    n_eval = 1200  # number of evaluations; n_eval = N from the paper.
    n_res = n_eval//40
    dt = (tmax-tmin)/n_eval  # step size; dt = Delta t from the paper.

    # generate the Kalman parameters corresponding to the prior
    prior = ibm_init(dt=dt,
                    n_order=n_order,
                    sigma=sigma)

    # instantiate the ODE solver

    # 5.  Evaluate the ODE solution
    # deterministic output: posterior mean
    mut, Sigmat = solve_mv(key = key,
                        fun = fitz,
                        x0 = x0,
                        theta = theta,
                        tmin = tmin,
                        tmax = tmax,
                        n_eval = n_eval,
                        wgt_meas = W,
                        **prior)

    # Compute exact solution
    tseq = np.linspace(0, 40, 41)
    ode0 = np.array([-1., 1.])
    exact = odeint(fitz0, ode0, tseq, args=(theta,))

    # Plot them
    plt.rcParams.update({'font.size': 40})
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    axs[0].plot(tseq, mut[::n_res, 0, 0], label = 'rodeo')
    axs[0].plot(tseq, exact[:, 0], label = 'True')
    axs[0].set_title("$V(t)$")
    
    axs[1].plot(tseq, mut[::n_res, 1, 0], label = 'rodeo')
    axs[1].plot(tseq, exact[:, 1], label = 'True')
    axs[1].set_title("$R(t)$")
    axs[1].legend(loc='upper left', bbox_to_anchor=[1, 1])
    fig.savefig('figures/fitzsolve.pdf')
