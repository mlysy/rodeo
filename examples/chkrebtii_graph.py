"""
Python file for the sole purpose of producing the figure in the chkrebtii example.
"""
import numpy as np
from math import sin, cos
import jax
import jax.numpy as jnp
from jax import random
from jax import config
import matplotlib.pyplot as plt

from rodeo import solve_sim, solve_mv
from rodeo.prior import ibm_init
from rodeo.interrogate import interrogate_chkrebtii
from euler import *
config.update("jax_enable_x64", True)
plt.rcParams.update({'font.size': 25, 'lines.linewidth':4})

# Example ODE Exact Solution for x_t^{(0)}
def ode_exact_x(t):
    return (-3*cos(t) + 2*sin(t) - sin(2*t))/3

# Example ODE Exact Solution for x_t^{(1)}
def ode_exact_x1(t):
    return (-2*cos(2*t) + 3*sin(t) + 2*cos(t))/3

# Example ode written for Euler Approximation
def ode_euler(x, t, theta):
    return jnp.array([x[1], jnp.sin(2*t) -x[0]])

# Setup the IVP problem in rodeo block form
def ode_rodeo(X_t, t):
    return jnp.array([[jnp.sin(2*t) - X_t[0, 0]]])

# jit functions
jit_mv = jax.jit(solve_mv, static_argnums=(1, 6, 7))
jit_sim = jax.jit(solve_sim, static_argnums=(1, 6, 7))

def solve(W, x0, t_min, t_max, n_steps, n_deriv, sigma, draws):
    """
    Calculates rodeo, euler, and exact solutions on the given grid for the chkrebtii ode.

    Args:
        W (ndarray(n_var, 1, p)): Corresponds to the :math:`W` matrix in the ODE equation.
        x0 (ndarray(n_var, p)): The initial value :math:`v`.
        
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points of the time interval that is evaluated, 
            such that discretization timestep is :math:`dt = (b-a)/N`.
        n_deriv (ndarray(n_var)): Number of derivatives of the ODE function.
        sigma (ndarray(n_var)): Scale parameter.
        draws (int): Number of samples we need to draw from the kalman solver.
    Returns:
        (tuple):
        - **tseq** (ndarray(n_steps+1)): Time discretization points for :math:`t = a,\ldots,b`.
        - **Xt** (ndarray(draws, n_steps+1, n_deriv)): Draws of the solution process :math:`X_t` at times
          :math:`t = a,\ldots,b`.
        - **x_euler** (ndarray(n_steps+1, 2)): Euler approximation of the solution process at
          times :math:`t = a,\ldots,b`.
        - **x_exact** (ndarray(n_steps+1, 2)): Exact solution at times :math:`t = a,\ldots,b`

    """
    dt = (t_max-t_min)/n_steps  # step size
    prior_Q, prior_R = ibm_init(
        dt=dt,
        n_deriv=n_deriv,
        sigma=sigma
    )
    # Run the solver which gives the posterior mean and variance
    key = jax.random.PRNGKey(0)  # PRNG key for JAX
    key, subkey = random.split(key)
    Xm, _ = jit_mv(
        key=subkey,
        # define ode
        ode_fun=ode_rodeo,
        ode_weight=W,
        ode_init=x0,
        t_min=t_min,
        t_max=t_max,
        # solver parameters
        n_steps=n_steps,
        interrogate=interrogate_chkrebtii,
        prior_weight=prior_Q,
        prior_var=prior_R
        
    )
    
    Xt = np.zeros((draws, n_steps+1, n_deriv))
    for i in range(draws):
        # Run the solver which gives a draw
        key, subkey = random.split(key)
        x_sol = jit_sim(
            key=subkey,
            # define ode
            ode_fun=ode_rodeo,
            ode_weight=W,
            ode_init=x0,
            t_min=t_min,
            t_max=t_max,
            # solver parameters
            n_steps=n_steps,
            interrogate=interrogate_chkrebtii,
            prior_weight=prior_Q,
            prior_var=prior_R
        )
        Xt[i] = x_sol[:, 0]
    x0_euler = x0.flatten()[:-2]
    x_euler = euler(ode_euler, x0_euler, None, t_min, t_max, n_steps)
    x_exact = np.zeros((n_steps+1, 2))
    tseq = np.linspace(t_min, t_max, n_steps+1)
    for i,t in enumerate(tseq):
        x_exact[i, 0] = ode_exact_x(t)
        x_exact[i, 1] = ode_exact_x1(t)

    return tseq, Xt, x_euler, x_exact, Xm[:, 0]

# Function that produces the graph as shown in README
def graph():
    """
    Produces the graph in Figure 1.

    """
    W = jnp.array([[[0.0, 0.0, 1.0, 0.0]]])   # ODE LHS matrix
    x0 = jnp.array([[-1., 0., 1., 0.0]])  # IVP initial value

    # time interval on which solution is sought
    t_min = 0.
    t_max = 10.

    # Define the solution prior process

    n_vars = 1  # number of system variables
    # number of continuous derivatives per variable
    n_deriv = 4
    sigma = jnp.array([.1] * n_vars)  # IBM process scale factor per variable

    # the prior parameters to the rodeo solver depend on solver step size
    N = [50, 100, 200]  # number of solver steps between tmin and tmax

    # Initialize variables for the graph
    N = [50, 100, 200]
    draws = 100
    dim_example = len(N)
    tseq = [None] * dim_example
    Xn = [None] * dim_example
    Xmean = [None] * dim_example
    x_euler = [None] * dim_example
    x_exact = [None] * dim_example

    for i in range(dim_example):
        tseq[i], Xn[i], x_euler[i], x_exact[i], Xmean[i] = solve(W=W,
                                                                x0=x0,
                                                                t_min=t_min, 
                                                                t_max=t_max, 
                                                                n_steps=N[i],
                                                                n_deriv=n_deriv,
                                                                sigma=sigma, 
                                                                draws=draws)

    
    fig, axs = plt.subplots(2, dim_example, figsize=(20, 10))
    for prow in range(2):
        for pcol in range(dim_example):
            # plot Kalman draws
            for i in range(draws):
                if i == (draws - 1):
                    axs[prow, pcol].plot(tseq[pcol], Xn[pcol][i,:,prow], 
                                        color="gray", alpha=.5, label="rodeo draws")
                else:
                    axs[prow, pcol].plot(tseq[pcol], Xn[pcol][i,:,prow], 
                                        color="gray", alpha=.3)
            # plot Kalman mean
            axs[prow, pcol].plot(tseq[pcol], Xmean[pcol][:,prow],
                                 label="rodeo mean")
            # plot Euler and Exact
            axs[prow, pcol].plot(tseq[pcol], x_euler[pcol][:,prow], 
                                label="Euler")
            axs[prow, pcol].plot(tseq[pcol], x_exact[pcol][:,prow], 
                                label="Exact")
            
            # set legend and title
            axs[prow, pcol].set_title("$x^{(%s)}(t)$;   $N=%s$" % (prow, N[pcol]))
            #axs[prow, pcol].set_ylabel("$x^{(%s)}_t$" % (prow))
    axs[0, -1].legend(loc='upper left', bbox_to_anchor=[1, 1])
    
    fig.tight_layout()
    fig.savefig('figures/chkrebtiifigure.pdf', bbox_inches='tight')
    plt.show()
    return fig

graph()
