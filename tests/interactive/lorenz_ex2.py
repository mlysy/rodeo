import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint
from rodeo.ode import interrogate_tronarp
from rodeo.fenrir import *
from rodeo.ibm import ibm_init
from jaxopt import ScipyMinimize
from pope import loglikehood
from jax.config import config
import matplotlib.pyplot as plt
config.update("jax_enable_x64", True)

def logprior(x, mean, sd):
    r"Calculate the loglikelihood of the normal distribution."
    return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

def double_nlpost(phi, x0):
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
    n_phi = len(phi_mean)
    theta = jnp.exp(phi[:n_phi])
    sigma = phi[-1]
    var_state = sigma**2*ode_init['var_state']
    lp = fenrir(key, lorenz, W_block, x0, theta, tmin, tmax, n_res,
                ode_init['trans_state'], ode_init['mean_state'], var_state,
                trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)
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
exact = odeint(lorenz0, ode0, tseq, args=(theta,), rtol=1e-20)
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

# grid for initial parameters
init_par = jnp.linspace(jnp.log(10e-2), jnp.log(10e1), 5)
lorenz_double = np.zeros((len(init_par), n_phi))
lorenz_fenrir = np.zeros((len(init_par), n_phi))
for i in range(len(init_par)):
    # parameter inference solver
    phi_init = jnp.array([init_par[i]]*n_var + [1])
    lorenz_double[i] = phi_fit(phi_init, x0_block, double_nlpost)
    lorenz_fenrir[i] = phi_fit(phi_init, x0_block, fenrir_nlpost)

# np.save("saves/lorenz_doubleip.npy", lorenz_double)
# # np.save("saves/lorenz_fenririp.npy", lorenz_fenrir)
# lorenz_double =np.load("saves/lorenz_doubleip.npy")
# lorenz_fenrir = np.load("saves/lorenz_fenririp.npy")

plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(n_var, figsize=(10, 10))
ylabel = [r"$\sigma$", r"$\rho$", r"$\beta$"]
for i in range(n_var):
    l2 = axs[i].scatter(jnp.exp(init_par), lorenz_double[:, i], label="double")
    l3 = axs[i].scatter(jnp.exp(init_par), lorenz_fenrir[:, i], label="fenrir")
    l4 = axs[i].axhline(theta_true[i], linestyle="--", color = "black")
    axs[i].set_xscale("log")
    axs[i].set(ylabel=ylabel[i])
handles = [l2, l3, l4]

fig.subplots_adjust(bottom=0.1, wspace=0.33)
axs[2].legend(handles = [l2,l3,l4] , labels=['double', 'fenrir', 'True'], loc='upper center', 
              bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=3)
# fig.savefig('figures/lorenzip.pdf')
