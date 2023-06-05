import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.ode import interrogate_tronarp
from rodeo.ibm import ibm_init
import rodeo.fenrir as ff
import rodeo.dalton as df
from jaxopt import ScipyMinimize
from jax import jacfwd, jacrev
import warnings
warnings.filterwarnings('ignore')
from diffrax import diffeqsolve, Dopri8, ODETerm, PIDController, ConstantStepSize, SaveAt
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
n_obs = 20
tseq = np.linspace(tmin, tmax, n_obs+1)
term = ODETerm(rax_fun)
stepsize = PIDController(rtol=1e-10, atol=1e-20)
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

# step size and parameters for solver prior
n_res = 200
sigma = 5e7
sigma = jnp.array([sigma]*n_var)
n_steps = n_res*n_obs
dt = (tmax-tmin)/n_steps
ode_init = ibm_init(dt, n_order, sigma)
key = jax.random.PRNGKey(0)

# dalton filter
dalton_m, _ = df.solve_mv(key, lorenz, W_block, x0_block, theta, tmin, tmax, n_res,
                          ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
                          trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)

# fenrir
fenrir_m, _ = ff.fenrir_mv(key, lorenz, W_block, x0_block, theta, tmin, tmax, n_res,
                           ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
                           trans_obs, mean_obs, var_obs, y_obs, interrogate=interrogate_tronarp)

# exact solution
tseq_sim = np.linspace(tmin, tmax, n_steps+1)
saveat = SaveAt(ts=tseq_sim)
exact_sim = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = .001,
            y0=jnp.array(ode0), max_steps = None, saveat=saveat, stepsize_controller=stepsize).ys

plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(n_var, figsize=(20, 10))
ylabel = [r'$x(t)$', r'$y(t)$', r'$z(t)$']

for i in range(n_var):
    l1, = axs[i].plot(tseq_sim, dalton_m[:, i, 0], label="DALTON", linewidth=4)
    l2, = axs[i].plot(tseq_sim, fenrir_m[:, i, 0], label="Fenrir", linewidth=4)
    l3, = axs[i].plot(tseq_sim, exact_sim[:, i], label='True', linewidth=2, color="black")
    l4 = axs[i].scatter(tseq, obs[:, i], label='Obs', color='red', s=40, zorder=4)
    axs[i].set(ylabel=ylabel[i])
handles = [l1, l2, l3, l4]

fig.subplots_adjust(bottom=0.1, wspace=0.33)

axs[2].legend(handles = [l1,l2,l3,l4] , labels=['DALTON', 'Fenrir', 'True', 'obs'], loc='upper center', 
              bbox_to_anchor=(0.5, -0.2),fancybox=False, shadow=False, ncol=4)

fig.savefig('figures/lorenzode.pdf')

# # ------------------------------------------------ parameter inference -------------------------------------------

def logprior(x, mean, sd):
    r"Calculate the loglikelihood of the normal distribution."
    return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

def dalton_nlpost(phi, x0):
    r"Compute the negative loglikihood of :math:`Y_t` using a DALTON."
    n_phi = len(phi_mean)
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
    r"Compute the negative loglikihood of :math:`Y_t` using Fenrir."
    n_phi = len(phi_mean)
    x0 = x0.at[:, 0].set(phi[n_theta:n_phi])
    theta = jnp.exp(phi[:n_theta])
    v0 = lorenz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    sigma = phi[-1]
    var_state = sigma**2*ode_init['var_state']
    lp = ff.fenrir(key, lorenz, W_block, x0, theta, tmin, tmax, n_res,
                ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
                trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def diffrax_nlpost(phi, x0):
    r"Compute the negative loglikihood of :math:`Y_t` using a deterministic solver."
    n_phi = len(phi_mean)
    theta = jnp.exp(phi[:n_theta])
    x0 = x0.at[:, 0].set(phi[n_theta:n_phi])
    x0 = x0.flatten()
    X_t = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = 0.01, 
                      y0=x0, saveat=saveat, stepsize_controller=stepsize,
                      max_steps = 1000000).ys
    lp = logprior(obs, X_t, gamma)
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

# parameters needed for diffrax
saveat = SaveAt(ts = tseq)
stepsize = ConstantStepSize()

# diffrax
key = jax.random.PRNGKey(0)
phi_init = jnp.append(jnp.log(theta_true), ode0)
phi_init = jnp.append(phi_init, jnp.array([1]))
phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((3,1)), diffrax_nlpost)
theta_diffrax = phi_sample(phi_hat, phi_var, n_samples)
theta_diffrax[:, :n_theta] = np.exp(theta_diffrax[:, :n_theta])

# posterior for dalton filter and fenrir 
# run for each res
n_reslst = [200, 400, 800]
sigmalst = [5e7, 1.5e8, 5e8]
theta_fenrir = np.zeros((len(n_reslst), n_samples, 6))
theta_dalton = np.zeros((len(n_reslst), n_samples, 6))
for i, n_res in enumerate(n_reslst):
    sigma = jnp.array([sigmalst[i]]*n_var)
    n_steps = n_obs*n_res
    dt = (tmax-tmin)/n_steps
    ode_init = ibm_init(dt, n_order, sigma)
    phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((3,1)), dalton_nlpost)
    theta_dalton[i] = phi_sample(phi_hat, phi_var, n_samples)
    theta_dalton[i, :, :n_theta] = np.exp(theta_dalton[i, :, :n_theta])
    phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((3,1)), fenrir_nlpost)
    theta_fenrir[i] = phi_sample(phi_hat, phi_var, n_samples)
    theta_fenrir[i, :, :n_theta] = np.exp(theta_fenrir[i, :, :n_theta])

# np.save("saves/lorenz_double3.npy", theta_dalton)
# np.save("saves/lorenz_fenrir3.npy", theta_fenrir)
# np.save("saves/lorenz_diffrax3.npy", theta_diffrax)
# theta_dalton = np.load("saves/lorenz_double3.npy")
# theta_fenrir = np.load("saves/lorenz_fenrir3.npy")
# theta_diffrax = np.load("saves/lorenz_diffrax3.npy")

# plot
plt.rcParams.update({'font.size': 20})
var_names = var_names = [r"$\alpha$", r"$\rho$", r"$\beta$", r"$x(0)$", r"$y(0)$", r"$z(0)$"]
meth_names = ["DALTON", "Fenrir", "RK"]
param_true = np.append(theta_true, ode0)
hlist = [1/200, 1/400, 1/800]
figure = theta_plotwd(theta_dalton, theta_fenrir, theta_diffrax, param_true, hlist, var_names, meth_names)
figure.savefig('figures/lorenzpost.pdf')
