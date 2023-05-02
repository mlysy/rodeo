import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint
from rodeo.ode import interrogate_tronarp
from rodeo.fenrir import *
from rodeo.ibm import ibm_init
from jaxopt import ScipyMinimize
from jax import jacfwd, jacrev
from pope import loglikehood_nn
from jax.config import config

import matplotlib.pyplot as plt
from theta_plot import theta_plotsingle
config.update("jax_enable_x64", True)
import warnings
from jax.config import config
warnings.filterwarnings('ignore')
from diffrax import diffeqsolve, Dopri8, ODETerm, SaveAt, ConstantStepSize

def logprior(x, mean, sd):
    r"Calculate the loglikelihood of the normal distribution."
    return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

def g(x_t, y_t):
    "Likelihood for diffrax"
    b0 = 0.1
    b1 = 0.5
    return jnp.sum(jsp.stats.poisson.logpmf(y_t, jnp.exp(b0+b1*x_t)))

def g2(x_obs, y_curr):
    "Observation function"
    n_block = y_curr.shape[0]
    b0 = 0.1
    b1 = 0.5
    return jnp.sum(jax.vmap(lambda b: jsp.stats.poisson.logpmf(y_curr[b], jnp.exp(b0+b1*x_obs[b])))(jnp.arange(n_block)))

def double_nlpost(phi, x0):
    n_phi = len(phi_mean)
    x0 = x0.at[:, 0].set(phi[n_theta:])
    theta = jnp.exp(phi[:n_theta])
    v0 = fitz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    lp = loglikehood_nn(key, fitz, W, x0, theta, tmin, tmax, n_res,
            ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
            g2, trans_obs, y_obs, interrogate_tronarp)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def diffrax_nlpost(phi, x0):
    r"Compute the negative loglikihood of :math:`Y_t` using a deterministic solver."
    n_phi = len(phi_mean)
    x0 = x0.at[:, 0].set(phi[n_theta:])
    x0 = x0.flatten()
    theta = jnp.exp(phi[:n_theta])
    X_t = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = diff_dt0, 
                        y0=jnp.array(x0), saveat=saveat, stepsize_controller=stepsize_controller).ys
    lp = g(X_t, y_obs.reshape((41, -1)))
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp
    
def phi_fit(phi_init, x0, obj_fun):
    r"""Compute the optimized :math:`\log{\theta}` and its variance given 
        :math:`Y_t`."""
    
    n_phi = len(phi_init)
    hes = jacfwd(jacrev(obj_fun))
    solver = ScipyMinimize(method="Newton-CG", fun = obj_fun)
    opt_res = solver.run(phi_init, x0)
    phi_hat = opt_res.params
    phi_fisher = hes(phi_hat, x0)
    phi_var = jsp.linalg.solve(phi_fisher, jnp.eye(n_phi))
    return phi_hat, phi_var
    
def phi_sample(phi_hat, phi_var, n_samples):
    r"""Simulate :math:`\theta` given the :math:`\log{\hat{\theta}}` 
        and its variance."""
    phi = np.random.default_rng(12345).multivariate_normal(phi_hat, phi_var, n_samples)
    return phi

def fitz0(X_t, t, theta):
    a, b, c = theta
    V, R = X_t 
    return np.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

def fitz(X_t, t, theta):
    "Fitz ODE written for jax"
    a, b, c = theta
    V, R = X_t[:, 0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                      [-1/c*(V - a + b*R)]])

def rax_fun(t, X_t, theta):
    "Fitz ODE written for diffrax"
    a, b, c = theta
    p = len(X_t)//2
    V, R = X_t[0], X_t[p]
    return jnp.array([c*(V - V*V*V/3 + R),
                        -1/c*(V - a + b*R)])

# problem setup and intialization
n_deriv = 3  # Total state
n_var = 2  # Total measures
n_dim = 2

# it is assumed that the solution is sought on the interval [tmin, tmax].
tmin = 0.
tmax = 40.
theta = np.array([0.2, 0.2, 3])

# Initial x0 for odeint
ode0 = np.array([-1., 1.])

# observations
n_obs = 40
tseq1 = np.linspace(tmin, tmax, n_obs+1)
exact1 = odeint(fitz0, ode0, tseq1, args=(theta,))
b0 = 0.1
b1 = 0.5
obs = np.random.default_rng(0).poisson(lam = jnp.exp(b0+b1*exact1))
trans_obs = jnp.array([[[1., 0., 0.]], [[1., 0., 0.]]])
y_obs = jnp.expand_dims(obs, -1)

# logprior parameters
theta_true = jnp.array([0.2, 0.2, 3]) # True theta
n_phi = 5
phi_mean = jnp.zeros(n_phi)
phi_sd = jnp.log(10)*jnp.ones(n_phi) 
n_theta = 3
n_samples = 100000

# parameters needed for diffrax
saveat = SaveAt(ts = tseq1)
term = ODETerm(rax_fun)
stepsize_controller = ConstantStepSize()
solver = Dopri8()
diff_dt0 = .1

# posteriors for diffrax
phi_init = jnp.append(jnp.log(theta_true)-0.01, ode0)
phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((2,1)), diffrax_nlpost)
theta_diffrax = phi_sample(phi_hat, phi_var, n_samples)
theta_diffrax[:, :n_theta] = np.exp(theta_diffrax[:, :n_theta])

key = jax.random.PRNGKey(0)
W = jnp.array([[[0., 1., 0.]], [[0., 1., 0.]]])  # ODE LHS matrix

# run for each res
n_reslst = [5, 10]
sigma = .1
sigma = jnp.array([sigma]*n_var)
n_order = jnp.array([n_deriv]*n_var)
theta_double = np.zeros((len(n_reslst), n_samples, 5))
for i, n_res in enumerate(n_reslst):
    n_steps = n_obs*n_res
    dt = (tmax-tmin)/n_steps
    ode_init = ibm_init(dt, n_order, sigma)
    phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((2,1)), double_nlpost)
    theta_double[i] = phi_sample(phi_hat, phi_var, n_samples)
    theta_double[i, :, :n_theta] = np.exp(theta_double[i, :, :n_theta])
    
# np.save("saves/fitzps_diffrax.npy", theta_diffrax)
# theta_diffrax = np.load("saves/fitzps_diffrax.npy")
# np.save("saves/fitzps_double2.npy", theta_double)

plt.rcParams.update({'font.size': 20})
var_names = ['a', 'b', 'c', r"$V(0)$", r"$R(0)$"]
param_true = np.append(theta_true, np.array([-1, 1]))
hlist = [1/5, 1/10]
figure = theta_plotsingle(theta_double, theta_diffrax, param_true, hlist, var_names, clip=[None, (0, 0.5), None, None, None], rows=1)
# figure.savefig('figures/fitzpoisfigure.pdf')
