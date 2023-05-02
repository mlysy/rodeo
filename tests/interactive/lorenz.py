import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.integrate import odeint
from rodeo.ode import interrogate_tronarp
from rodeo.ibm import ibm_init
from rodeo.fenrir import fenrir
import pope as df
import fenrir_filter as ff
from jaxopt import ScipyMinimize
from jax import jacfwd, jacrev
from jax.config import config

import matplotlib.pyplot as plt
from theta_plot import theta_plotwd
config.update("jax_enable_x64", True)

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
n_obs = 20
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

# step size and parameters for solver prior
n_res = 200
sigma = 5e7
sigma = jnp.array([sigma]*n_var)
n_steps = n_res*n_obs
dt = (tmax-tmin)/n_steps
ode_init = ibm_init(dt, n_order, sigma)
key = jax.random.PRNGKey(0)

# double filter
double_m, _ = df.solve_mv(key, lorenz, W_block, x0_block, theta, tmin, tmax, n_res,
                          ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
                          trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)

# fenrir
fenrir_m, _ = ff.fenrir_filter(key, lorenz, W_block, x0_block, theta, tmin, tmax, n_res,
                               ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
                               trans_obs, mean_obs, var_obs, y_obs, interrogate=interrogate_tronarp)

# exact solution
tseq_sim = np.linspace(tmin, tmax, n_steps+1)
exact_sim = odeint(lorenz0, ode0, tseq_sim, args=(theta,), rtol=1e-20)

plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(n_var, figsize=(20, 10))
ylabel = ['x', 'y', 'z']

for i in range(n_var):
    l1, = axs[i].plot(tseq_sim, exact_sim[:, i], label='True')
    l2, = axs[i].plot(tseq_sim, double_m[:, i, 0], label="double")
    l3, = axs[i].plot(tseq_sim, fenrir_m[:, i, 0], label="fenrir")
    l4 = axs[i].scatter(tseq, obs[:, i], label='Obs', color='red')
    axs[i].set(ylabel=ylabel[i])
handles = [l1, l2, l3, l4]

fig.subplots_adjust(bottom=0.1, wspace=0.33)

axs[2].legend(handles = [l1,l2,l3,l4] , labels=['True', 'double', 'fenrir', 'obs'], loc='upper center', 
              bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=4)

# fig.savefig('figures/lorenzode.pdf')

# ------------------------------------------------ parameter inference -------------------------------------------

def logprior(x, mean, sd):
    r"Calculate the loglikelihood of the normal distribution."
    return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

def double_nlpost(phi, x0):
    n_phi = len(phi_mean)
    # x0 = x0_initialize(phi[n_theta:], x0)
    x0 = x0.at[:, 0].set(phi[n_theta:n_phi])
    theta = jnp.exp(phi[:n_theta])
    v0 = lorenz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    sigma = phi[-1]
    var_state = sigma**2*ode_init['var_state']
    lp = df.loglikehood(key, lorenz, W_block, x0, theta, tmin, tmax, n_res,
            ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
            trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def fenrir_nlpost(phi, x0):
    n_phi = len(phi_mean)
    # x0 = x0_initialize(phi[n_theta:], x0)
    x0 = x0.at[:, 0].set(phi[n_theta:n_phi])
    theta = jnp.exp(phi[:n_theta])
    v0 = lorenz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    sigma = phi[-1]
    var_state = sigma**2*ode_init['var_state']
    lp = fenrir(key, lorenz, W_block, x0, theta, tmin, tmax, n_res,
                ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
                trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def phi_fit(phi_init, x0, obj_fun):
    r"""Compute the optimized :math:`\log{\theta}` and its variance given 
        :math:`Y_t`."""
    
    n_phi = len(phi_init) - 1
    hes = jacfwd(jacrev(obj_fun))
    solver = ScipyMinimize(method="Newton-CG", fun = obj_fun)
    opt_res = solver.run(phi_init, x0)
    phi_hat = opt_res.params[:n_phi]
    phi_fisher = hes(phi_hat, x0)
    phi_var = jsp.linalg.solve(phi_fisher, jnp.eye(n_phi))
    return phi_hat, phi_var
    
def phi_sample(phi_hat, phi_var, n_samples):
    r"""Simulate :math:`\theta` given the :math:`\log{\hat{\theta}}` 
        and its variance."""
    phi = np.random.default_rng(12345).multivariate_normal(phi_hat, phi_var, n_samples)
    return phi

# logprior parameters
theta_true = jnp.array([28, 10, 8/3]) # True theta
n_phi = 6
phi_mean = jnp.zeros(n_phi)
phi_sd = jnp.log(10)*jnp.ones(n_phi) 
n_theta = 3
n_samples = 100000

# define gm prior
n_res = 200
sigma = 5e7
sigma = jnp.array([sigma]*n_var)
n_steps = n_res*n_obs
dt = (tmax-tmin)/n_steps
ode_init = ibm_init(dt, n_order, sigma)
key = jax.random.PRNGKey(0)

# posterior for double filter and fenrir 
phi_init = jnp.append(jnp.log(theta_true)-0.01, ode0)
phi_init = jnp.append(phi_init, jnp.array([1]))
theta_fenrir = np.zeros((1, n_samples, 6))
theta_double = np.zeros((1, n_samples, 6))
phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((3,1)), double_nlpost)
theta_double[0] = phi_sample(phi_hat, phi_var, n_samples)
theta_double[0, :, :n_theta] = np.exp(theta_double[0, :, :n_theta])
phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((3,1)), fenrir_nlpost)
theta_fenrir[0] = phi_sample(phi_hat, phi_var, n_samples)
theta_fenrir[0, :, :n_theta] = np.exp(theta_fenrir[0, :, :n_theta])

# np.save("saves/lorenz_double.npy", theta_double)
# np.save("saves/lorenz_fenrir.npy", theta_fenrir)
# theta_double = np.load("saves/lorenz_double.npy")
# theta_fenrir = np.load("saves/lorenz_fenrir.npy")

# plot
plt.rcParams.update({'font.size': 20})
var_names = var_names = [r"$\sigma$", r"$\rho$", r"$\beta$", "x", "y", "z"]
param_true = np.append(theta_true, ode0)
hlist = [1/200]
figure = theta_plotwd(theta_double, theta_fenrir, param_true, hlist, var_names)

# figure.savefig('figures/lorenzpost.pdf')
