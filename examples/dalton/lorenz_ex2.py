import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.ode import interrogate_tronarp
from rodeo.fenrir import fenrir
from rodeo.ibm import ibm_init
from rodeo.dalton import loglikehood
from jaxopt import ScipyMinimize
import warnings
warnings.filterwarnings('ignore')
from diffrax import diffeqsolve, Dopri8, ODETerm, PIDController, ConstantStepSize, SaveAt
from jax.config import config
import matplotlib.pyplot as plt
config.update("jax_enable_x64", True)

def logprior(x, mean, sd):
    r"Calculate the loglikelihood of the normal distribution."
    return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

def dalton_nlpost(phi, x0):
    r"Compute the negative loglikihood of :math:`Y_t` using a DALTON."
    n_phi = len(phi_mean)
    theta = jnp.exp(phi[:n_phi])
    sigma = phi[-1]
    var_state = sigma**2*ode_init['var_state']
    lp = loglikehood(key, lorenz, W_block, x0, theta, tmin, tmax, n_res,
            ode_init['trans_state'], ode_init['mean_state'], var_state,
            trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def fenrir_nlpost(phi, x0):
    r"Compute the negative loglikihood of :math:`Y_t` using Fenrir."
    n_phi = len(phi_mean)
    theta = jnp.exp(phi[:n_phi])
    sigma = phi[-1]
    var_state = sigma**2*ode_init['var_state']
    lp = fenrir(key, lorenz, W_block, x0, theta, tmin, tmax, n_res,
                ode_init['trans_state'], ode_init['mean_state'], var_state,
                trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def diffrax_nlpost(phi, x0):
    r"Compute the negative loglikihood of :math:`Y_t` using a deterministic solver."
    n_phi = len(phi_mean)
    theta = jnp.exp(phi[:n_phi])
    # theta = jnp.exp(phi)
    X_t = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = 0.01, 
                      y0=x0, saveat=saveat, stepsize_controller=stepsize,
                      max_steps = 1000000).ys
    lp = logprior(obs, X_t, gamma)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def phi_fit(phi_init, x0, obj_fun):
    solver = ScipyMinimize(method="Newton-CG", fun = obj_fun)
    opt_res = solver.run(phi_init, x0)
    phi_hat = opt_res.params[:n_phi]
    return jnp.exp(phi_hat)

# ODE function
def lorenz(X_t, t, theta):
    rho, sigma, beta = theta
    x, y, z = X_t[:,0]
    dx = -sigma*x + sigma*y
    dy = rho*x - y -x*z
    dz = -beta*z + x*y
    return jnp.array([[dx], [dy], [dz]])

def lorenz0(X_t, t, theta):
    rho, sigma, beta = theta
    x, y, z = X_t
    dx = -sigma*x + sigma*y
    dy = rho*x - y -x*z
    dz = -beta*z + x*y
    return np.array([dx, dy, dz])

def rax_fun(t, X_t, theta):
    rho, sigma, beta = theta
    x, y, z = X_t
    dx = -sigma*x + sigma*y
    dy = rho*x - y -x*z
    dz = -beta*z + x*y
    return jnp.array([dx, dy, dz])

# problem setup and intialization
n_deriv = 3  # Total state; q
n_var = 3  # Total variables

# Time interval on which a solution is sought.
tmin = 0.
tmax = 20.
theta = jnp.array([28, 10, 8/3])

# Initial W for jax block
W_mat = np.zeros((n_var, 1, n_deriv))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

# Initial x0 for jax block
x0_block = jnp.array([[-12., 70., 0.], [-5., 125, 0.], [38., -124/3, 0.]])
n_order = jnp.array([n_deriv]*n_var)

# Initial x0 for odeint
ode0 = jnp.array([-12., -5., 38.])

# Get observations
n_obs = 200
tseq = np.linspace(tmin, tmax, n_obs+1)
term = ODETerm(rax_fun)
stepsize = PIDController(rtol=1e-20, atol=1e-10)
solver = Dopri8()
# Get exact solutions for the Lorenz System
saveat = SaveAt(ts=tseq)
exact = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = .001,
                  y0=ode0, max_steps = None, saveat=saveat, stepsize_controller=stepsize).ys
gamma = np.sqrt(0.005)
e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=exact.shape)
obs = exact + gamma*e_t
y_obs = jnp.expand_dims(obs, -1)

# arguments involving the observations for solvers
mean_obs = jnp.zeros((n_var, 1))
trans_obs = np.zeros((n_var, 1, n_deriv))
trans_obs[:, :, 0] = 1
trans_obs = jnp.array(trans_obs)
var_obs = gamma**2*jnp.ones((n_var, 1, 1))

# logprior parameters
theta_true = jnp.array([28, 10, 8/3]) # True theta
n_phi = 3
phi_mean = jnp.zeros(n_phi)
phi_sd = jnp.log(10)*jnp.ones(n_phi) 

# step size and parameters for solver prior
n_res = 200
sigma = 5e7
sigma = jnp.array([sigma]*n_var)
n_steps = n_res*n_obs
dt = (tmax-tmin)/n_steps
ode_init = ibm_init(dt, n_order, sigma)
key = jax.random.PRNGKey(0)

# parameters needed for diffrax
saveat = SaveAt(ts = tseq)
term = ODETerm(rax_fun)
stepsize = ConstantStepSize()
solver = Dopri8()

# grid for initial parameters
init_par = jnp.linspace(jnp.log(10e-2), jnp.log(10e1), 5)
lorenz_dalton = np.zeros((len(init_par), n_phi))
lorenz_fenrir = np.zeros((len(init_par), n_phi))
lorenz_diffrax = np.zeros((len(init_par), n_phi))
for i in range(len(init_par)):
    # parameter inference solver
    phi_init = jnp.array([init_par[i]]*n_var + [1])
    lorenz_dalton[i] = phi_fit(phi_init, x0_block, dalton_nlpost)
    lorenz_fenrir[i] = phi_fit(phi_init, x0_block, fenrir_nlpost)
    lorenz_diffrax[i] = phi_fit(phi_init, ode0, diffrax_nlpost)

# np.save("saves/lorenz_doubleip.npy", lorenz_dalton)
# np.save("saves/lorenz_fenririp.npy", lorenz_fenrir)
# np.save("saves/lorenz_diffraxip.npy", lorenz_diffrax)
# lorenz_dalton = np.load("saves/lorenz_doubleip2.npy")
# lorenz_fenrir = np.load("saves/lorenz_fenririp2.npy")
# lorenz_diffrax = np.load("saves/lorenz_diffraxip2.npy")

plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(ncols=n_var, figsize=(20, 5))
title = [r"$\alpha$", r"$\rho$", r"$\beta$"]
for i in range(n_var):
    l1 = axs[i].scatter(jnp.exp(init_par), lorenz_dalton[:, i], label="DALTON", s=60)
    l2 = axs[i].scatter(jnp.exp(init_par), lorenz_fenrir[:, i], label="Fenrir", s=60)
    l3 = axs[i].scatter(jnp.exp(init_par), lorenz_diffrax[:, i], label="RK", s=60)
    l4 = axs[i].axhline(theta_true[i], linestyle="--", color = "black")
    axs[i].set_xscale("log")
    if i == 1:
        axs[i].set_yscale("log")
    axs[i].set_title(title[i])
axs[0].set_ylabel("Optimized Parameter")
fig.supxlabel("Initial Parameter", y=0.15)
handles = [l1, l2, l3, l4]
fig.subplots_adjust(bottom=.3, wspace=0.33)
fig.legend(handles = handles , labels=['DALTON', 'Fenrir', 'RK', 'True'], loc="lower center",
           fancybox=False, shadow=False, ncol=4)
fig.savefig('figures/lorenzip.pdf')
