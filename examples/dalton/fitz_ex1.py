import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.ode import interrogate_tronarp
from rodeo.fenrir import *
from rodeo.ibm import ibm_init
from jaxopt import ScipyMinimize
from jax import jacfwd, jacrev
from rodeo.dalton import loglikehood
from jax.config import config

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from theta_plot import theta_plot
config.update("jax_enable_x64", True)
import warnings
warnings.filterwarnings('ignore')
from diffrax import diffeqsolve, Dopri8, ODETerm, PIDController, SaveAt

backend = mpl.get_backend()
mpl.use('agg')

def logprior(x, mean, sd):
    r"Calculate the loglikelihood of the normal distribution."
    return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

def dalton_nlpost(phi, x0):
    r"Compute the negative loglikihood of :math:`Y_t` using DALTON."
    n_phi = len(phi_mean)
    x0 = x0.at[:, 0].set(phi[n_theta:n_phi])
    theta = jnp.exp(phi[:n_theta])
    v0 = fitz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    sigma = phi[-1]
    var_state = sigma**2*ode_init['var_state']
    lp = loglikehood(key, fitz, W, x0, theta, tmin, tmax, n_res,
            ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
            trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def fenrir_nlpost(phi, x0):
    r"Compute the negative loglikihood of :math:`Y_t` using Fenrir."
    n_phi = len(phi_mean)
    x0 = x0.at[:, 0].set(phi[n_theta:n_phi])
    theta = jnp.exp(phi[:n_theta])
    v0 = fitz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    sigma = phi[-1]
    var_state = sigma**2*ode_init['var_state']
    lp = fenrir(key, fitz, W, x0, theta, tmin, tmax, n_res,
                ode_init['trans_state'], ode_init['mean_state'], ode_init['var_state'],
                trans_obs, mean_obs, var_obs, y_obs, interrogate_tronarp)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def diffrax_nlpost(phi, x0):
    r"Compute the negative loglikihood of :math:`Y_t` using a deterministic solver."
    n_phi = len(phi_mean)
    # x0 = x0_initialize(phi[n_theta:], x0)
    x0 = x0.at[:, 0].set(phi[n_theta:n_phi])
    x0 = x0.flatten()
    theta = jnp.exp(phi[:n_theta])
    X_t = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = diff_dt0, 
                        y0=jnp.array(x0), saveat=saveat, stepsize_controller=stepsize_controller).ys
    # lp = g(X_t, y_obs.reshape((41, -1)))
    lp = logprior(y_obs.reshape((41, -1)), X_t, gamma)
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
n_steps = 200
tmin = 0.
tmax = 40.
theta = jnp.array([0.2, 0.2, 3])

# Initial x0 for odeint
ode0 = jnp.array([-1., 1.])

# parameters needed for diffrax
term = ODETerm(rax_fun)
stepsize_controller = PIDController(rtol=1e-10, atol=1e-8)
solver = Dopri8()
diff_dt0 = .1

# observations
n_obs = 40
tseq = np.linspace(tmin, tmax, n_obs+1)
saveat = SaveAt(ts = tseq)
exact = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = diff_dt0, 
                  y0=ode0, saveat=saveat, stepsize_controller=stepsize_controller).ys
gamma = np.sqrt(.005)
e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=exact.shape)
obs = exact + gamma*e_t
y_obs = jnp.expand_dims(obs, -1)
trans_obs = jnp.array([[[1., 0., 0.]], [[1., 0., 0.]]])
mean_obs = jnp.zeros((2, 1))
var_obs = gamma**2*jnp.array([[[1.]],[[1.]]])

# plot one graph
plt.rcParams.update({'font.size': 30})
fig1, axs = plt.subplots(2, 1, figsize=(8, 10))
axs[0].plot(tseq, exact[:,0], label = 'True', linewidth=4)
axs[0].scatter(tseq, obs[:,0], label = 'Obs', color='orange', s=200, zorder=2)
axs[0].set_title("$V(t)$")
axs[1].plot(tseq, exact[:,1], label = 'True', linewidth=4)
axs[1].scatter(tseq, obs[:,1], label = 'Obs', color='orange', s=200, zorder=2)
axs[1].set_title("$R(t)$")
fig1.tight_layout()
fig1.suptitle('(a)', horizontalalignment='left', x=0, y=1)
fig1.savefig('figures/fitzODEvert.pdf')

# logprior parameters
theta_true = jnp.array([0.2, 0.2, 3]) # True theta
n_phi = 5
phi_mean = jnp.zeros(n_phi)
phi_sd = jnp.log(10)*jnp.ones(n_phi) 
n_theta = 3
n_samples = 100000

# # posteriors for diffrax
phi_init = jnp.append(jnp.log(theta_true), ode0)
phi_init = jnp.append(phi_init, jnp.array([1]))
phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((2,1)), diffrax_nlpost)
theta_diffrax = phi_sample(phi_hat, phi_var, n_samples)
theta_diffrax[:, :n_theta] = np.exp(theta_diffrax[:, :n_theta])
key = jax.random.PRNGKey(0)
W = jnp.array([[[0., 1., 0.]], [[0., 1., 0.]]])  # ODE LHS matrix

# run for each res
n_reslst = [4, 10, 20]
sigma = 10
sigma = jnp.array([sigma]*n_var)
n_order = jnp.array([n_deriv]*n_var)
theta_fenrir = np.zeros((len(n_reslst), n_samples, 5))
theta_dalton = np.zeros((len(n_reslst), n_samples, 5))
for i, n_res in enumerate(n_reslst):
    n_steps = n_obs*n_res
    dt = (tmax-tmin)/n_steps
    ode_init = ibm_init(dt, n_order, sigma)
    phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((2,1)), dalton_nlpost)
    theta_dalton[i] = phi_sample(phi_hat, phi_var, n_samples)
    theta_dalton[i, :, :n_theta] = np.exp(theta_dalton[i, :, :n_theta])
    phi_hat, phi_var = phi_fit(phi_init, jnp.zeros((2,1)), fenrir_nlpost)
    theta_fenrir[i] = phi_sample(phi_hat, phi_var, n_samples)
    theta_fenrir[i, :, :n_theta] = np.exp(theta_fenrir[i, :, :n_theta])
    
# np.save("saves/theta_diffraxg.npy", theta_diffrax)
# np.save("saves/theta_fenrirg.npy", theta_fenrir)
# np.save("saves/theta_doubleg.npy", theta_dalton)
# theta_diffrax = np.load("saves/theta_diffraxg.npy")
# theta_fenrir = np.load("saves/theta_fenrirg.npy")
# theta_dalton = np.load("saves/theta_doubleg.npy")

plt.rcParams.update({'font.size': 30})
var_names = ['a', 'b', 'c', r"$V(0)$", r"$R(0)$"]
meth_names = ["DALTON", "Fenrir"]
param_true = np.append(theta_true, np.array([-1, 1]))
hlist = [1/4, 1/10, 1/20]
fig2 = theta_plot(theta_dalton, theta_fenrir, theta_diffrax, param_true, hlist, var_names, meth_names, clip=[None, (0, 4), None, None, None], rows=1)
fig2.suptitle('(b)', horizontalalignment='left', x=0, y=1)
fig2.savefig('figures/fitzfigure.pdf')

# combine figures
c1 = fig1.canvas
c2 = fig2.canvas
c1.draw()
c2.draw()
a1 = np.array(c1.buffer_rgba())
a2 = np.array(c2.buffer_rgba())
a = np.hstack((a1,a2))
mpl.use(backend)
fig,ax = plt.subplots(figsize=(28, 10))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_axis_off()
ax.matshow(a)
fig.savefig('figures/fitzcombined.pdf')
