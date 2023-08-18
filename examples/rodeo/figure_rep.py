r"""
This module is for replicating the figures in the paper.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
from functools import partial
from jaxopt import ScipyMinimize

# configs
import warnings
from jax.config import config
warnings.filterwarnings('ignore')
config.update("jax_enable_x64", True)

# solvers and IBM prior
from rodeo.ibm import ibm_init
from rodeo.ode import solve_mv, solve_sim, interrogate_chkrebtii, interrogate_kramer
from rodeo.fenrir import fenrir
from rodeo.dalton import dalton, daltonng
from rodeo.oc_mcmc import oc_mcmc
from diffrax import diffeqsolve, Dopri5, ODETerm, PIDController, SaveAt, DirectAdjoint
from blackjax import hmc, window_adaptation

# plot
from math import ceil
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
plt.rcParams.update({'font.size': 20})

# plotting function --------------------------------------------------------

def clip_ext(x, min, max):
    r"""Clip extreme values"""
    return x[(x>min) & (x<max)]

def theta_plot(theta, theta_names, theta_diffrax, theta_true, step_sizes, var_names, clip=None, rows=1):
    r"""Plot the distribution of :math:`\theta` using various approximations."""
    n_hlst, _, n_theta = theta[0].shape
    ncol = ceil(n_theta/rows) +1
    nrow = len(theta)
    fig = plt.figure(figsize=(20, 5*ceil(nrow/2)*rows))
    patches = [None]*(n_hlst+2)
    if clip is None:
        clip = [None]*n_theta
    carry = 0
    for t in range(1,n_theta+1):
        row = (t-1)//(ncol-1)*(nrow-1)
        if t%ncol == 0:
            carry += 1
        axes = []
        for r in range(nrow):
            axes.append(fig.add_subplot(rows*nrow, ncol, t+(row+r)*(ncol)+carry))
            if r > 0:
                axes[r].get_shared_x_axes().join(axes[r], axes[0])
            if r < nrow-1:
                axes[r].set_xticks([])

            if (t+carry)%ncol==1:
                axes[r].set_ylabel(theta_names[r])

        axes[0].set_title(var_names[t-1])
        for i, axs in enumerate(axes):
            axs.axvline(x=theta_true[t-1], linewidth=1, color='r', linestyle='dashed')
            axs.set_yticks([])

        for h in range(n_hlst):
            if t==1:
                patches[h] = mpatches.Patch(color='C{}'.format(h), label='$\\Delta$ t ={}'.format(step_sizes[h]))    
            for r in range(nrow):
                tmp_data = theta[r][h, :, t-1]
                if clip[t-1] is not None:
                    tmp_data = clip_ext(tmp_data, clip[t-1][0], clip[t-1][1])
                sns.kdeplot(tmp_data, ax=axes[r])
        
        for r in range(nrow):
            tmp_data = theta_diffrax[:, t-1]
            if clip[t-1] is not None:
                tmp_data = clip_ext(tmp_data, clip[t-1][0], clip[t-1][1])
            sns.kdeplot(tmp_data, ax=axes[r],  color='black')
        if t==n_theta:
            patches[-2] = mpatches.Patch(color='black', label="True Posterior")
            patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\theta$')
            
    fig.legend(handles=patches, framealpha=0.5, loc=7)
    
    fig.tight_layout()
    plt.show()
    return fig

# general framework --------------------------------------------------------
def x0_initialize(phi, x0):
    r"""
    Initialize x0 for none missing initial values
    
    Args:
        phi : The initial values for the missing x0.
        x0 : The initial value which may contain some missing values.
    
    Returns:
        x0 : x0 filled with missing initial values.
    """
    j = 0
    for ind in mask:
        x0 = x0.at[ind,0].set(phi[j])
        j+=1
    return x0

def constrain_pars(phi, x0):
    r"""
    Separate parameters into ODE parameters, initial values and tuning parameters.

    Args:
        phi : Parameters to optimize over.
        x0 : The initial value which may contain some missing values.

    Returns:
        (tuple):
        - **theta** : ODE parameters.
        - **x0** : Initial values.
        - **sigma** : Tuning parameters.
    """
    theta = jnp.exp(phi[:n_theta])
    x0 = x0_initialize(phi[n_theta:n_phi], x0)
    sigma = phi[n_phi:]
    return theta, x0, sigma

def jaxopt_solver(fun, method="Newton-CG"):
    r"""
    Jaxopt solver or equivalent.

    Args:
        fun : Objective function to optimize.
        method : Choice of optimization method.

    Returns:
        (solver): Jaxopt solver.
    """
    return ScipyMinimize(fun=fun, method=method, jit=True)

def bna_fit(key, fun, n_samples, phi_init, x0):
    """
    Sample from the Bayesian normal approximation.

    Args:
        key : PRNG key.
        fun : Objective function to optimize.
        n_samples : Number of samples.
        phi_init : Parameters to optimize over.
        x0 : The initial value which may contain some missing values.
    
    Returns:
        phi : Sample phi.
    """
    solver = jaxopt_solver(fun)
    hes = jax.jacfwd(jax.jacrev(fun))
    opt_res = solver.run(phi_init, x0)
    phi_hat = opt_res.params
    phi_fisher = hes(phi_hat, x0)
    phi_var = jsp.linalg.solve(phi_fisher[:n_phi, :n_phi], jnp.eye(n_phi))
    phi = jax.random.multivariate_normal(
        key=key, mean=phi_hat[:n_phi], cov=phi_var, shape=(n_samples, ))
    # phi = np.random.default_rng(12345).multivariate_normal(phi_hat[:n_phi], phi_var, n_samples)
    return phi

# specific to each example -------------------------------------------------
# euler solver
# @partial(jax.jit, static_argnums=(0, 5))
# def euler(fun, x0, theta, tmin, tmax, n_steps):
#     r"""
#     Euler approximation of the ODE solution.

#     Args:
#         fun : Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
#         x0 : Initial value of the state variable :math:`X_t` at time :math:`t = a`.
#         theta : Parameters in the ODE function.
#         tmin : First time point of the time interval to be evaluated; :math:`a`.
#         tmax : Last time point of the time interval to be evaluated; :math:`b`.
#         n_steps : Number of discretization points (:math:`N`) of the time interval that is evaluated, 
#             such that discretization timestep is :math:`dt = (b-a)/N`.
        
#     Returns:
#         (ndarray(n_steps+1, n_state)): Euler approximation of the ODE solution.
#     """
#     step_size = (tmax - tmin)/n_steps
#     def scan_fun(x_old, t):
#         x_new = x_old + fun(x_old, tmin + step_size*t, theta)*step_size
#         return x_new, x_new
#     (_, X_t) = lax.scan(scan_fun, x0, jnp.arange(n_steps))

#     X_t = jnp.concatenate([x0[None], X_t])
#     return X_t

# # Fitz-Hugh model ----------------------------------------------------------
# def fitz_euler(Xt, t, theta):
#     r"Fitz-Hugh Nagumo ODE written for Euler"
#     a, b, c = theta
#     V, R = Xt 
#     return jnp.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

# def fitz(Xt, t, theta):
#     "Fitz ODE written for rodeo/chkrebtii/fenrir/dalton"
#     a, b, c = theta
#     V, R = Xt[:, 0]
#     return jnp.array([[c*(V - V*V*V/3 + R)],
#                       [-1/c*(V - a + b*R)]])

# def fitz_rax(t, Xt, theta):
#     "Fitz ODE written for diffrax"
#     a, b, c = theta
#     p = len(Xt)//2
#     V, R = Xt[0], Xt[p]
#     return jnp.array([c*(V - V*V*V/3 + R),
#                         -1/c*(V - a + b*R)])

# # --- data simulation ------------------------------------------------------
# # Produce a Pseudo-RNG key
# key = jax.random.PRNGKey(101)
# key, *subkeys = jax.random.split(key, num=3)  # split keys to not reuse

# W = jnp.array([[[0., 1., 0.]], [[0., 1., 0.]]])  # LHS matrix of ODE
# x0 = jnp.array([[-1., 1., 0.], [1., 1 / 3, 0.]])  # initial value for the IVP
# theta = jnp.array([.2, .2, 3])  # ODE parameters

# # Time interval on which a solution is sought.
# tmin = 0.
# tmax = 40.
# dt_obs = 1  # Time between observations

# # Define the prior process
# n_vars = 2
# n_deriv = jnp.array([3] * n_vars)

# # IBM process scale factor
# sigma = jnp.array([.01] * n_vars)

# # use large number for accurate solution
# # higher resolution means smaller step size
# n_res = 100  # resolution number;
# n_steps = int(n_res * (tmax - tmin) / dt_obs)
# dt = (tmax - tmin) / n_steps  # step size

# # generate the Kalman parameters corresponding to the prior
# prior_pars = ibm_init(dt=dt,
#                       n_deriv=n_deriv,
#                       sigma=sigma)

# # deterministic output: posterior mean
# mut, Sigmat = solve_mv(key=subkeys[0],
#                        # define ode
#                        fun=fitz,
#                        W=W,
#                        x0=x0,
#                        theta=theta,
#                        tmin=tmin,
#                        tmax=tmax,
#                        # solver parameters
#                        n_steps=n_steps,
#                        interrogate=interrogate_kramer,
#                        **prior_pars)

# # generate observations
# noise_sd = 0.2  # Standard deviation in noise model
# Xt = mut[::n_res, :, 0]
# et = jax.random.normal(key=subkeys[1], shape=Xt.shape)
# Yt = Xt + noise_sd * et

# # # plot observations
# tseq = np.linspace(tmin, tmax, len(Yt))
# fig, axs = plt.subplots(1, 2, figsize=(20, 5))
# axs[0].plot(tseq, Xt[:,0], label = 'X_t')
# axs[0].scatter(tseq, Yt[:,0], label = 'Y_t', color='orange')
# axs[0].set_title("$V(t)$")
# axs[1].plot(tseq, Xt[:,1], label = 'X_t')
# axs[1].scatter(tseq, Yt[:,1], label = 'Y_t', color='orange')
# axs[1].set_title("$R(t)$")
# axs[1].legend(loc='upper left', bbox_to_anchor=[1, 1])
# fig.savefig('figures/fitzsim.pdf')

# # --- parameter inference ------------------------------------------------
# n_phi = 5  # number of parameters + initial values to estimate
# phi_mean = jnp.zeros((n_phi,))
# phi_sd = jnp.log(10) * jnp.ones((n_phi,))
# n_theta = 3
# n_samples = 100000

# # parameters needed for diffrax
# term = ODETerm(fitz_rax)
# solver = Dopri5()
# diff_dt0 = 1
# saveat = SaveAt(ts=tseq)
# stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

# key, key2 = jax.random.split(key)

# def fitz_logpost_diffrax(phi, x0):
#     r"""
#     Compute the logposterior for the diffrax solver in the Fitz-Hugh Nagumo ODE.
#     Sames arguments as `fitz_logpost_basic`.
#     """
#     theta, x0, sigma = constrain_pars(phi, x0)
#     x0 = x0.flatten()
#     Xt = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = diff_dt0, 
#                      y0=jnp.array(x0), saveat=saveat, stepsize_controller=stepsize_controller,
#                      adjoint=DirectAdjoint()).ys
#     loglik = jnp.sum(jsp.stats.norm.logpdf(Yt, Xt, noise_sd))
#     logprior = jnp.sum(jsp.stats.norm.logpdf(phi[:n_phi], phi_mean, phi_sd))
#     return -(loglik + logprior)

# def fitz_logpost_basic(phi, x0):
#     r"""
#     Compute the logposterior for the basic approximation in the Fitz-Hugh Nagumo ODE.
    
#     Args:
#         phi : Parameters to optimize over; 
#             `phi = (log a, log b, log c, V0, R0, sigma_V, sigma_R)`.
#         x0 : The initial value which may contain some missing values.
    
#     Returns:
#         (float): Logposterior approximation.
#     """
#     theta, x0, sigma = constrain_pars(phi, x0)
#     v0 = fitz(x0, 0, theta)
#     x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
#     var_state = jax.vmap(lambda b: sigma[b]**2*prior_pars['var_state'][b])(jnp.arange(2))
#     Xt, _ = solve_mv(
#         key=key,
#         fun=fitz,
#         W=W,
#         x0=x0,
#         theta=theta,
#         tmin=tmin,
#         tmax=tmax,
#         interrogate=interrogate_kramer,
#         n_steps=n_steps,
#         wgt_state=prior_pars['wgt_state'],
#         var_state=var_state
#     )
#     # compute the loglikelihood and the log-prior
#     loglik = jnp.sum(jsp.stats.norm.logpdf(
#         x=Yt,
#         loc=Xt[::n_res, :, 0],  # thin solver output
#         scale=noise_sd
#     ))
#     logprior = jnp.sum(jsp.stats.norm.logpdf(
#         x=phi[:n_phi],
#         loc=phi_mean,
#         scale=phi_sd
#     ))
#     return -(loglik + logprior)

# def fitz_logpost_euler(phi, x0):
#     r"""
#     Compute the logposterior for the Euler solver in the Fitz-Hugh Nagumo ODE.
#     Sames arguments as `fitz_logpost_basic`.
#     """
#     theta, x0, sigma = constrain_pars(phi, x0)
#     x0 = x0.flatten()
#     Xt = euler(fitz_euler, x0, theta, tmin, tmax, n_steps)[::n_res]
#     loglik = jnp.sum(jsp.stats.norm.logpdf(Yt, Xt, noise_sd))
#     logprior = jnp.sum(jsp.stats.norm.logpdf(phi[:n_phi], phi_mean, phi_sd))
#     return -(loglik + logprior)

# # fenrir/dalton extra parameters
# trans_obs = jnp.zeros((41, n_vars, 1, n_deriv[0]))
# trans_obs = trans_obs.at[:].set(jnp.array([[[1., 0., 0.]], [[1., 0., 0.]]]))
# var_obs = noise_sd**2*jnp.array([[[1.]],[[1.]]])

# def fitz_logpost_fenrir(phi, x0):
#     r"""
#     Compute the logposterior for the fenrir solver in the Fitz-Hugh Nagumo ODE.
#     Sames arguments as `fitz_logpost_basic`.
#     """
#     theta, x0, sigma = constrain_pars(phi, x0)
#     v0 = fitz(x0, 0, theta)
#     x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
#     var_state = jax.vmap(lambda b: sigma[b]**2*prior_pars['var_state'][b])(jnp.arange(2))
#     loglik = fenrir(
#         key=key,
#         fun=fitz,
#         W=W,
#         x0=x0,
#         theta=theta,
#         tmin=tmin,
#         tmax=tmax,
#         interrogate=interrogate_kramer,
#         n_res=n_res,
#         wgt_state=prior_pars['wgt_state'],
#         var_state=var_state,
#         trans_obs=trans_obs,
#         var_obs=var_obs,
#         y_obs=jnp.expand_dims(Yt, -1)

#     )
#     logprior = jnp.sum(jsp.stats.norm.logpdf(phi[:n_phi], phi_mean, phi_sd))
#     return -(loglik + logprior)

# def fitz_logpost_dalton(phi, x0):
#     r"""
#     Compute the logposterior for the dalton solver in the Fitz-Hugh Nagumo ODE.
#     Sames arguments as `fitz_logpost_basic`.
#     """
#     theta, x0, sigma = constrain_pars(phi, x0)
#     v0 = fitz(x0, 0, theta)
#     x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
#     var_state = jax.vmap(lambda b: sigma[b]**2*prior_pars['var_state'][b])(jnp.arange(2))
#     loglik = dalton(
#         key=key,
#         fun=fitz,
#         W=W,
#         x0=x0,
#         theta=theta,
#         tmin=tmin,
#         tmax=tmax,
#         interrogate=interrogate_kramer,
#         n_res=n_res,
#         wgt_state=prior_pars['wgt_state'],
#         var_state=var_state,
#         # var_state=prior_pars['var_state'],
#         trans_obs=trans_obs,
#         var_obs=var_obs,
#         y_obs=jnp.expand_dims(Yt, -1)

#     )
#     logprior = jnp.sum(jsp.stats.norm.logpdf(phi[:n_phi], phi_mean, phi_sd))
#     return -(loglik + logprior)

# # chkrebtii mcmc sampler
# class fitz_ocmcmc(oc_mcmc):
#     def __init__(self, fun, W, tmin, tmax, phi_mean, phi_sd, Yt, x0, noise_sd):
#         # some values need to be determined in the for-loop
#         super().__init__(fun, W, None, tmin, tmax, None, None, None, Yt) 
#         self.phi_mean = phi_mean
#         self.phi_sd = phi_sd
#         self.x0 = x0
#         self.noise_sd = noise_sd

#     def logprior(self, phi):
#         r"Calculate the loglikelihood of the prior."
#         return jnp.sum(jsp.stats.norm.logpdf(x=phi[:len(self.phi_mean)], loc=self.phi_mean, scale=self.phi_sd))

#     def loglik(self, X_t):
#         r"Calculate the loglikelihood of the observations."
#         return jnp.sum(jsp.stats.norm.logpdf(x=self.y_obs, loc=X_t, scale=self.noise_sd))

#     def solve(self, key, phi):
#         r"Solve the ODE given the theta"
#         theta, x0, sigma = constrain_pars(phi, self.x0)
#         v0 = fitz(x0, 0, theta)
#         x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
#         var_state = jax.vmap(lambda b: sigma[b]**2*prior_pars['var_state'][b])(jnp.arange(2))
#         Xt = solve_sim(
#             key=key,
#             fun=fitz,
#             W=W,
#             x0=x0,
#             theta=theta,
#             tmin=tmin,
#             tmax=tmax,
#             interrogate=interrogate_chkrebtii,
#             n_steps=n_steps,
#             wgt_state=prior_pars['wgt_state'],
#             var_state=var_state
#         )
#         Xt = Xt[::self.n_res, :, 0]
#         return Xt
    
#     def mcmc_sample(self, key, phi_init, n_samples):
#         r"""
#         Sample via the MCMC algorithm using Chkbretii solver.

#         Args:
#             key : PRNG key.
#             phi_init : Initial parameter to be optimized.
#             n_samples : Number of samples to return.
        
#         Return:
#             phi : Samples to return.
#         """
#         param = jnp.diag(jnp.array([0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]))
#         key, subkey = jax.random.split(key)
#         initial_state = self.init(subkey, phi_init)
#         def one_step(state, key):
#             state, sample = self.step(key, state, param)
#             return state, sample

#         keys = jax.jax.random.split(key, n_samples)
#         _, samples = jax.lax.scan(one_step, initial_state, keys)
#         return samples['theta'] # need to convert back to normal scale afterwards

# # blackjax sampler
# def inference_loop(key, kernel, initial_state, n_samples):
#     r"""
#     Sample via the Blackjax MCMC algorithm.

#     Args:
#         key : PRNG key.
#         kernel : Type of MCMC algorithm.
#         initial_state : Starting state for the algorithm; use warmup to find this.
#         n_samples: Number of samples to return.
    
#     Return:
#         phi : Samples to return.
#     """
#     @jax.jit
#     def one_step(state, rng_key):
#         state, _ = kernel(rng_key, state)
#         return state, state

#     keys = jax.random.split(key, n_samples)
#     _, states = jax.lax.scan(one_step, initial_state, keys)
#     return states

# # initial parameters
# ode0 = jnp.array([-1., 1.])
# x0 = jnp.zeros((2,1)) 
# mask = range(2) # estimate both initial values
# phi_init = jnp.append(jnp.log(theta), ode0)
# phi_init = jnp.append(phi_init, jnp.ones(n_vars))

# # blackjax hmc parameters
# fitz_logpost_bj = lambda phi: -fitz_logpost_basic(phi, x0=x0)
# num_integration_steps = 5
# step_size = 1e-2

# # blackjax sampling setup
# n_res = 10  # resolution number;
# n_steps = int(n_res * (tmax - tmin) / dt_obs)
# dt = (tmax - tmin) / n_steps  # step size
# prior_pars = ibm_init(dt=dt,
#                       n_deriv=n_deriv,
#                       sigma=sigma)
# warmup = window_adaptation(hmc, fitz_logpost_bj, num_integration_steps=num_integration_steps)
# initial_state, kernel, _ = warmup.run(key2, phi_init)

# # diffrax posterior
# # phi_diffrax = bna_fit(key2, fitz_logpost_diffrax, n_samples, phi_init, x0)
# # theta_diffrax = jnp.concatenate([jnp.exp(phi_diffrax[:,:n_theta]), phi_diffrax[:, n_theta:]], axis=1)

# # # # solver posteriors
# n_res_list = np.array([5,10,20,50])
# theta_euler = np.zeros((len(n_res_list), n_samples, 5))
# # theta_basic = np.zeros((len(n_res_list), n_samples, 5))
# # theta_fenrir = np.zeros((len(n_res_list), n_samples, 5))
# # theta_dalton = np.zeros((len(n_res_list), n_samples, 5))
# # # theta_ch = np.zeros((len(n_res_list), n_samples, n_phi))
# # # fitz_ch = fitz_ocmcmc(fitz, W, tmin, tmax, phi_mean, phi_sd, Yt, x0, noise_sd)
# theta_hmc = np.zeros((len(n_res_list), n_samples//10, 5))
# for i, n_res in enumerate(n_res_list):
#     # prior setup for solvers
#     n_steps = int(n_res * (tmax - tmin) / dt_obs)
#     dt = (tmax - tmin) / n_steps  # step size
#     prior_pars = ibm_init(dt, n_deriv, sigma=sigma)
#     # euler approximation
#     theta_euler[i] = bna_fit(key2, fitz_logpost_euler, n_samples, phi_init, x0)
#     theta_euler[i, :, :n_theta] = np.exp(theta_euler[i, :, :n_theta])
#     # basic approximation
#     # theta_basic[i] = bna_fit(key2, fitz_logpost_basic, n_samples, phi_init, x0)
#     # theta_basic[i, :, :n_theta] = np.exp(theta_basic[i, :, :n_theta])
#     # # fenrir
#     # theta_fenrir[i] = bna_fit(key2, fitz_logpost_fenrir, n_samples, phi_init, x0)
#     # theta_fenrir[i, :, :n_theta] = np.exp(theta_fenrir[i, :, :n_theta])
#     # # dalton
#     # theta_dalton[i] = bna_fit(key2, fitz_logpost_dalton, n_samples, phi_init, x0)
#     # theta_dalton[i, :, :n_theta] = np.exp(theta_dalton[i, :, :n_theta])
#     # chkrebtii
#     # fitz_ch.n_steps = n_steps
#     # fitz_ch.n_res = n_res_list[i]
#     # fitz_ch.prior_pars = prior_pars
#     # theta_ch[i] = fitz_ch.mcmc_sample(key2, phi_init, n_samples)[:, :n_phi]
#     # theta_ch[i, :, :n_theta] = np.exp(theta_ch[i, :, :n_theta])
#     # blackjax hmc
#     # warmup = window_adaptation(hmc, fitz_logpost_bj, num_integration_steps=num_integration_steps)
#     # initial_state, kernel, _ = warmup.run(key2, phi_init)
#     theta_hmc[i] = inference_loop(key2, kernel, initial_state, n_samples//10).position[:, :n_phi]
#     theta_hmc[i, :, :n_theta] = np.exp(theta_hmc[i, :, :n_theta])

# # np.save('saves/fitz_theta_diffrax.npy', theta_diffrax)
# # np.save('saves/fitz_theta_basic.npy', theta_basic)
# # np.save('saves/fitz_theta_fenrir.npy', theta_fenrir)
# # np.save('saves/fitz_theta_dalton.npy', theta_dalton)
# # np.save('saves/fitz_theta_ch.npy', theta_ch)
# np.save('saves/fitz_theta_hmc.npy', theta_hmc)

# # # # --- plot ---------------------------------------------------------------
# theta_diffrax = np.load("saves/fitz_theta_diffrax.npy")
# theta_basic = np.load("saves/fitz_theta_basic.npy")
# theta_fenrir = np.load("saves/fitz_theta_fenrir.npy")
# theta_dalton = np.load("saves/fitz_theta_dalton.npy")
# theta_ch = np.load("saves/fitz_theta_ch.npy")
# theta_hmc = np.load("saves/fitz_theta_hmc.npy")
# plot_theta = [theta_euler, theta_basic, theta_fenrir, theta_dalton, theta_ch, theta_hmc]
# theta_names = ["Euler", "basic", "fenrir", "dalton", "chkrebtii", "hmc"]
# var_names = ['a', 'b', 'c', r"$V(0)$", r"$R(0)$"]
# param_true = np.append(theta, np.array([-1, 1]))
# clip = [None, (0.0, 0.5), None, None, None]
# figure = theta_plot(plot_theta, theta_names, theta_diffrax, param_true, 1/n_res_list, var_names, clip=clip, rows=1)
# figure.savefig('figures/fitzfigure2.pdf')

# # Hes1 model ---------------------------------------------------------------
# def hes1(Xt, t, theta):
#     "Hes1 model on the log-scale"
#     P, M, H = jnp.exp(Xt[:, 0])
#     a, b, c, d, e, f, g = theta
    
#     x1 = -a*H + b*M/P - c
#     x2 = -d + e/(1+P*P)/M
#     x3 = -a*P + f/(1+P*P)/H - g
#     return jnp.array([[x1], [x2], [x3]])

# def hes1_rax(t, Xt, theta):
#     "Hes1 ODE written for diffrax"
#     a, b, c, d, e, f, g = theta
#     P, M, H = jnp.exp(Xt)
#     a, b, c, d, e, f, g = theta
#     x1 = -a*H + b*M/P - c
#     x2 = -d + e/(1+P*P)/M
#     x3 = -a*P + f/(1+P*P)/H - g
#     return jnp.array([x1, x2, x3])

# # # --- data simulation ------------------------------------------------------
# # Produce a Pseudo-RNG key
# def hes1(Xt, t, theta):
#     "Hes1 model on the log-scale"
#     P, M, H = jnp.exp(Xt[:, 0])
#     a, b, c, d, e, f, g = theta
    
#     x1 = -a*H + b*M/P - c
#     x2 = -d + e/(1+P*P)/M
#     x3 = -a*P + f/(1+P*P)/H - g
#     return jnp.array([[x1], [x2], [x3]])

# def hes1_rax(t, Xt, theta):
#     "Hes1 ODE written for diffrax"
#     a, b, c, d, e, f, g = theta
#     P, M, H = jnp.exp(Xt)
#     a, b, c, d, e, f, g = theta
#     x1 = -a*H + b*M/P - c
#     x2 = -d + e/(1+P*P)/M
#     x3 = -a*P + f/(1+P*P)/H - g
#     return jnp.array([x1, x2, x3])

# # # --- data simulation ------------------------------------------------------
# # Produce a Pseudo-RNG key
# key = jax.random.PRNGKey(1000)
# key, *subkeys = jax.random.split(key, num=3)  # split keys to not reuse

# # Time interval on which a solution is sought.
# tmin = 0.
# tmax = 240.
# dt_obs = 7.5  # Time between observations

# # Define the prior process
# n_vars = 3  # number of system variables
# n_deriv = jnp.array([3] * n_vars)

# # IBM process scale factor
# sigma = jnp.array([.1] * n_vars)

# theta = jnp.array([0.022, 0.3, 0.031, 0.028, 0.5, 20, 0.3])  # ODE parameters
# W_mat = np.zeros((n_vars, 1, 3))
# W_mat[:, :, 1] = 1
# W = jnp.array(W_mat)  # LHS matrix of ODE
# x0 = jnp.log(jnp.array([[1.439], [2.037], [17.904]]))
# v0 = hes1(x0, 0, theta)
# x0 = jnp.concatenate([x0, v0, jnp.zeros((3,1))], axis=1) # initial value for the IVP

# # use large number for accurate solution
# # higher resolution means smaller step size
# n_res = 100  # resolution number;
# n_steps = int(n_res * (tmax - tmin) / dt_obs)
# dt = (tmax - tmin) / n_steps  # step size

# # generate the Kalman parameters corresponding to the prior
# prior_pars = ibm_init(dt=dt,
#                       n_deriv=n_deriv,
#                       sigma=sigma)

# # deterministic output: posterior mean
# mut, Sigmat = solve_mv(key=subkeys[0],
#                        # define ode
#                        fun=hes1,
#                        W=W,
#                        x0=x0,
#                        theta=theta,
#                        tmin=tmin,
#                        tmax=tmax,
#                        # solver parameters
#                        n_steps=n_steps,
#                        interrogate=interrogate_kramer,
#                        **prior_pars)

# # generate observations
# noise_sd = 0.15  # Standard deviation in noise model
# Xt = mut[::n_res, :, 0]

# def hes1_obs(sol):
#     r"Given the solution process, get the corresponding observations"
#     Xt = jnp.zeros((len(sol)))
#     Xt = Xt.at[::2].set(sol[::2, 0])
#     Xt = Xt.at[1::2].set(sol[1::2, 1])
#     return Xt

# # observations for diffrax/basic
# Xt = hes1_obs(Xt)
# et = jax.random.normal(key=subkeys[1], shape=Xt.shape)
# Yt = Xt + noise_sd * et

# # fenrir observations format
# Yt2 = np.zeros((len(Yt), n_deriv[0]))
# Yt2[::2, 0] = Yt[::2]
# Yt2[1::2, 1] = Yt[1::2]
# Yt2 = jnp.array(Yt2)

# # --- parameter inference ------------------------------------------------
# n_phi = 10  # number of parameters + initial values to estimate
# n_theta = 7
# phi_mean = jnp.zeros((n_phi,))
# phi_sd = jnp.log(10) * jnp.ones((n_phi,))
# n_samples = 100000

# # parameters needed for diffrax
# term = ODETerm(hes1_rax)
# solver = Dopri5()
# diff_dt0 = .1
# tseq = np.linspace(tmin, tmax, len(Yt))
# saveat = SaveAt(ts=tseq)
# stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

# key, key2 = jax.random.split(key)
# def hes1_logpost_diffrax(phi, x0):
#     r"""
#     Compute the logposterior for the diffrax solver in the Hes1 ODE.
#     Sames arguments as `fitz_logpost_basic`.
#     """
#     theta, x0, sigma = constrain_pars(phi, x0)
#     x0 = x0.flatten()
#     Xt = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = diff_dt0, 
#                      y0=jnp.array(x0), saveat=saveat, stepsize_controller=stepsize_controller,
#                      adjoint=DirectAdjoint()).ys
#     Xt = hes1_obs(Xt)
#     loglik = jnp.sum(jsp.stats.norm.logpdf(Yt, Xt, noise_sd))
#     logprior = jnp.sum(jsp.stats.norm.logpdf(phi[:n_phi], phi_mean, phi_sd))
#     return -(loglik + logprior)

# def hes1_logpost_basic(phi, x0):
#     r"""
#     Compute the logposterior for the basic approximation in the Hes1 ODE.
    
#     Args:
#         phi : Parameters to optimize over; 
#             `phi = (log a, log b, log c, log d, log e, log f, log g, 
#                     P0, M0, H0, sigma_P, sigma_M, sigma_H)`.
#         x0 : The initial value which may contain some missing values.
    
#     Returns:
#         (float): Logposterior approximation.
#     """
#     theta, x0, sigma = constrain_pars(phi, x0)
#     v0 = hes1(x0, 0, theta)
#     X0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
#     var_state = jax.vmap(lambda b: sigma[b]**2*prior_pars['var_state'][b])(jnp.arange(3))
#     Xt, _ = solve_mv(
#         key=key,
#         fun=hes1,
#         W=W,
#         x0=X0,
#         theta=theta,
#         tmin=tmin,
#         tmax=tmax,
#         interrogate=interrogate_kramer,
#         n_steps=n_steps,
#         wgt_state=prior_pars['wgt_state'],
#         # var_state=prior_pars['var_state']
#         var_state=var_state
#     )
#     # compute the loglikelihood and the log-prior
#     Xt = hes1_obs(Xt[::n_res, :, 0])
#     loglik = jnp.sum(jsp.stats.norm.logpdf(
#         x=Yt,
#         loc=Xt,  # thin solver output
#         scale=noise_sd
#     ))
#     logprior = jnp.sum(jsp.stats.norm.logpdf(
#         x=phi[:n_phi],
#         loc=phi_mean,
#         scale=phi_sd
#     ))
#     return -(loglik + logprior)

# # fenrir extra parameters
# trans_obs = jnp.zeros((len(Yt), n_vars, 1, n_deriv[0]))
# trans_obs = trans_obs.at[::2].set(jnp.array([[[1., 0., 0.]], [[0., 0., 0.]], [[0., 0., 0.]]]))
# trans_obs = trans_obs.at[1::2].set(jnp.array([[[0., 0., 0.]], [[1., 0., 0.]], [[0., 0., 0.]]]))
# var_obs = noise_sd**2*jnp.array([[[1.]], [[1.]], [[1.]]])

# def hes1_logpost_fenrir(phi, x0):
#     r"""
#     Compute the logposterior for the fenrir solver in the Hes1 ODE.
    
#     Args:
#         phi : Parameters to optimize over; 
#             `phi = (log a, log b, log c, log d, log e, log f, log g, 
#                     P0, M0, H0, sigma_P, sigma_M, sigma_H)`.
#         x0 : The initial value which may contain some missing values.
    
#     Returns:
#         (float): Logposterior approximation.
#     """
#     theta, x0, sigma = constrain_pars(phi, x0)
#     v0 = hes1(x0, 0, theta)
#     x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
#     var_state = jax.vmap(lambda b: sigma[b]**2*prior_pars['var_state'][b])(jnp.arange(3))
#     loglik = fenrir(
#         key=key,
#         fun=hes1,
#         W=W,
#         x0=x0,
#         theta=theta,
#         tmin=tmin,
#         tmax=tmax,
#         interrogate=interrogate_kramer,
#         n_res=n_res,
#         wgt_state=prior_pars['wgt_state'],
#         var_state=var_state,
#         trans_obs=trans_obs,
#         var_obs=var_obs,
#         y_obs=jnp.expand_dims(Yt2, -1)

#     )
#     logprior = jnp.sum(jsp.stats.norm.logpdf(
#         x=phi[:n_phi],
#         loc=phi_mean,
#         scale=phi_sd
#     ))
#     return -(loglik + logprior)

# # initial parameters
# ode0 = jnp.log(np.array([1.439, 2.037, 17.904]))
# x0 = jnp.zeros((3,1)) 
# mask = range(3) # estimate both initial values
# phi_init = jnp.append(jnp.log(theta), ode0)
# phi_init = jnp.append(phi_init, jnp.ones(n_vars))

# # diffrax posterior
# # phi_diffrax = bna_fit(key2, hes1_logpost_diffrax, n_samples, phi_init, x0)
# # theta_diffrax = jnp.exp(phi_diffrax)

# # solver posteriors
# n_res_list = np.array([4, 6, 10])
# # theta_basic = np.zeros((len(n_res_list), n_samples, 10))
# # theta_fenrir = np.zeros((len(n_res_list), n_samples, 10))
# # for i, n_res in enumerate(n_res_list):
# #     # prior setup for solvers
# #     n_steps = int(n_res * (tmax - tmin) / dt_obs)
# #     dt = (tmax - tmin) / n_steps  # step size
# #     prior_pars = ibm_init(dt, n_deriv, sigma=sigma)
    
# #     # basic approximation
# #     theta_basic[i] = bna_fit(key2, hes1_logpost_basic, n_samples, phi_init, x0)
# #     theta_basic[i] = np.exp(theta_basic[i])

# #     # fenrir
# #     theta_fenrir[i] = bna_fit(key2, hes1_logpost_fenrir, n_samples, phi_init, x0)
# #     theta_fenrir[i] = np.exp(theta_fenrir[i])

# # np.save("saves/hes1_theta_diffrax.npy", theta_diffrax)
# # np.save("saves/hes1_theta_basic.npy", theta_basic)
# # np.save("saves/hes1_theta_fenrir.npy", theta_fenrir)

# # --- plot ---------------------------------------------------------------
# theta_diffrax = np.load("saves/hes1_theta_diffrax.npy")
# theta_basic = np.load("saves/hes1_theta_basic.npy")
# theta_fenrir = np.load("saves/hes1_theta_fenrir.npy")
# plot_theta = [theta_basic, theta_fenrir]
# theta_names = ["basic", "fenrir"]
# var_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', r"$P(0)$", r"$M(0)$", r"$H(0)$"]
# param_true = np.append(theta, np.exp(ode0))
# clip = [(0.00, 0.10), None, None, None, None, (0, 30), (0, 1), None, None, (0,40)]
# figure = theta_plot(plot_theta, theta_names, theta_diffrax, param_true, dt_obs/n_res_list, var_names, clip, rows=2)
# figure.savefig('figures/hes1figure.pdf')

# SEIRAH model -------------------------------------------------------------
def seirah(Xt, t, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = Xt[:, 0]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q= theta
    D_h = 30
    x1 = -b*S*(I + alpha*A)/N
    x2 = b*S*(I + alpha*A)/N - E/D_e
    x3 = r*E/D_e - I/D_q - I/D_I
    x4 = (I + A)/D_I + H/D_h
    x5 = (1-r)*E/D_e - A/D_I
    x6 = I/D_q - H/D_h
    return jnp.array([[x1], [x2], [x3], [x4], [x5], [x6]])

def seirah_rax(t, Xt, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = Xt
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q= theta
    D_h = 30
    x1 = -b*S*(I + alpha*A)/N
    x2 = b*S*(I + alpha*A)/N - E/D_e
    x3 = r*E/D_e - I/D_q - I/D_I
    x4 = (I + A)/D_I + H/D_h
    x5 = (1-r)*E/D_e - A/D_I
    x6 = I/D_q - H/D_h
    return jnp.array([x1, x2, x3, x4, x5, x6])

# --- data simulation ------------------------------------------------------
# Produce a Pseudo-RNG key
key = jax.random.PRNGKey(100)
key, *subkeys = jax.random.split(key, num=3)  # split keys to not reuse

# Time interval on which a solution is sought.
tmin = 0.
tmax = 60.
dt_obs = 1.0  # Time between observations

# Define the prior process
n_vars = 6  # number of system variables
n_deriv = jnp.array([3] * n_vars)

# IBM process scale factor
sigma = jnp.array([.1] * n_vars)

theta = jnp.array([2.23, 0.034, 0.55, 5.1, 2.3, 1.13]) # ODE parameters
W_mat = np.zeros((n_vars, 1, 3))
W_mat[:, :, 1] = 1
W = jnp.array(W_mat)  # LHS matrix of ODE
x0 = jnp.array([[63884630.], [15492.], [21752.], [0.], [618013.], [13388.]])
v0 = seirah(x0, 0, theta)
x0 = jnp.concatenate([x0, v0, jnp.zeros((6,1))], axis=1)

# use large number for accurate solution
# higher resolution means smaller step size
n_res = 100  # resolution number;
n_steps = int(n_res * (tmax - tmin) / dt_obs)
dt = (tmax - tmin) / n_steps  # step size

# generate the Kalman parameters corresponding to the prior
prior_pars = ibm_init(dt=dt,
                      n_deriv=n_deriv,
                      sigma=sigma)

# deterministic output: posterior mean
mut, Sigmat = solve_mv(key=subkeys[0],
                       # define ode
                       fun=seirah,
                       W=W,
                       x0=x0,
                       theta=theta,
                       tmin=tmin,
                       tmax=tmax,
                       # solver parameters
                       n_steps=n_steps,
                       interrogate=interrogate_kramer,
                       **prior_pars)

# generate observations
Xt = mut[::n_res, :, 0]

def covid_obs(Xt, theta):
    r"Compute the observations as detailed in the paper"
    I_in = theta[1]*Xt[:,1]/theta[3]
    H_in = Xt[:,2]/theta[5]
    Xin = jnp.array([I_in, H_in]).T
    return Xin

Xin = covid_obs(Xt, theta)
Yt = jax.random.poisson(key=subkeys[1], lam=Xin)

# --- parameter inference ------------------------------------------------
n_phi = 8  # number of parameters + initial values to estimate
phi_mean = jnp.zeros((n_phi,))
phi_sd = jnp.log(10) * jnp.ones((n_phi,))
n_theta = 6
n_samples = 100000

# parameters needed for diffrax
term = ODETerm(seirah_rax)
solver = Dopri5()
diff_dt0 = 1
tseq = np.linspace(tmin, tmax, len(Yt))
saveat = SaveAt(ts=tseq)
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

key, key2 = jax.random.split(key)
def seirah_logpost_diffrax(phi, x0):
    r"""
    Compute the logposterior for the diffrax solver in the SEIRAH ODE.
    Sames arguments as `fitz_logpost_basic`.
    """
    theta, x0, sigma = constrain_pars(phi, x0)
    x0 = x0.at[mask, 0].set(jnp.exp(x0[mask, 0]))
    x0 = x0.flatten()
    Xt = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0 = diff_dt0, 
                     y0=jnp.array(x0), saveat=saveat, stepsize_controller=stepsize_controller,
                     adjoint=DirectAdjoint()).ys
    Xin = covid_obs(Xt, theta)
    loglik = jnp.sum(jsp.stats.poisson.logpmf(Yt, Xin))
    logprior = jnp.sum(jsp.stats.norm.logpdf(phi[:n_phi], phi_mean, phi_sd))
    return -(loglik + logprior)

def seirah_logpost_basic(phi, x0):
    r"""
    Compute the logposterior for the basic approximation in the Hes1 ODE.
    
    Args:
        phi : Parameters to optimize over; 
            `phi = (log a, log b, log c, log d, log e, log f, log g, 
                    P0, M0, H0, sigma_P, sigma_M, sigma_H)`.
        x0 : The initial value which may contain some missing values.
    
    Returns:
        (float): Logposterior approximation.
    """
    theta, x0, sigma = constrain_pars(phi, x0)
    x0 = x0.at[mask, 0].set(jnp.exp(x0[mask, 0]))
    v0 = seirah(x0, 0, theta)
    X0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    var_state = jax.vmap(lambda b: sigma[b]**2*prior_pars['var_state'][b])(jnp.arange(6))
    Xt, _ = solve_mv(
        key=key,
        fun=seirah,
        W=W,
        x0=X0,
        theta=theta,
        tmin=tmin,
        tmax=tmax,
        interrogate=interrogate_kramer,
        n_steps=n_steps,
        wgt_state=prior_pars['wgt_state'],
        var_state=var_state
    )
    # compute the loglikelihood and the log-prior
    Xin = covid_obs(Xt[::n_res, :, 0], theta)
    loglik = jnp.sum(jsp.stats.poisson.logpmf(
        k=Yt, 
        mu=Xin
    ))
    logprior = jnp.sum(jsp.stats.norm.logpdf(
        x=phi[:n_phi],
        loc=phi_mean,
        scale=phi_sd
    ))
    return -(loglik + logprior)

# dalton extra parameters
trans_obs = jnp.zeros((len(Yt), n_vars, 1, n_deriv[0]))
trans_obs = trans_obs.at[:, 1:3, 0, 0].set(1)
def fun_obs(Xt, Yt, theta, i):
    r"Likelihood of SEIRAH observations for DALTON"
    n_yblock = Yt.shape[0]
    I_in = theta[1]*Xt[1]/theta[3]
    H_in = Xt[2]/theta[5]
    Xin = jnp.array([I_in, H_in])
    return jnp.sum(jax.vmap(lambda b: jsp.stats.poisson.logpmf(Yt[b], Xin[b]))(jnp.arange(n_yblock)))

def seirah_logpost_dalton(phi, x0):
    r"""
    Compute the logposterior for the dalton solver in the Hes1 ODE.
    
    Args:
        phi : Parameters to optimize over; 
            `phi = (log a, log b, log c, log d, log e, log f, log g, 
                    P0, M0, H0, sigma_P, sigma_M, sigma_H)`.
        x0 : The initial value which may contain some missing values.
    
    Returns:
        (float): Logposterior approximation.
    """
    theta, x0, sigma = constrain_pars(phi, x0)
    x0 = x0.at[mask, 0].set(jnp.exp(x0[mask, 0]))
    v0 = seirah(x0, 0, theta)
    X0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    var_state = jax.vmap(lambda b: sigma[b]**2*prior_pars['var_state'][b])(jnp.arange(6))
    loglik = daltonng(
        key=key,
        fun=seirah,
        W=W,
        x0=X0,
        theta=theta,
        tmin=tmin,
        tmax=tmax,
        interrogate=interrogate_kramer,
        n_res=n_res,
        wgt_state=prior_pars['wgt_state'],
        # var_state=prior_pars['var_state'],
        var_state =var_state,
        fun_obs=fun_obs,
        trans_obs=trans_obs,
        y_obs=jnp.expand_dims(Yt, -1)
    )
    logprior = jnp.sum(jsp.stats.norm.logpdf(
        x=phi[:n_phi],
        loc=phi_mean,
        scale=phi_sd
    ))
    return -(loglik + logprior)

# initial parameters
ode0 = jnp.array([63884630, 15492, 21752, 0, 618013, 13388])
x0 = jnp.array([[63884630.], [-1.], [-1.], [0.], [618013.], [13388.]]) # -1 for missing values
mask = jnp.array([1,2]) # estimate both initial values
phi_init = jnp.append(jnp.log(theta), jnp.log(ode0[mask]))
phi_init = jnp.append(phi_init, jnp.ones(n_vars))

# diffrax posterior
# phi_diffrax = bna_fit(key2, seirah_logpost_diffrax, n_samples, phi_init, x0)
# theta_diffrax = np.exp(phi_diffrax)

# solver posteriors
n_res_list = np.array([5, 10, 20, 50])
theta_basic = np.zeros((len(n_res_list), n_samples, 8))
theta_dalton = np.zeros((len(n_res_list), n_samples, 8))
for i, n_res in enumerate(n_res_list):
    # prior setup for solvers
    n_steps = int(n_res * (tmax - tmin) / dt_obs)
    dt = (tmax - tmin) / n_steps  # step size
    prior_pars = ibm_init(dt, n_deriv, sigma=sigma)
    
    # basic approximation
    theta_basic[i] = bna_fit(key2, seirah_logpost_basic, n_samples, phi_init, x0)
    theta_basic[i] = np.exp(theta_basic[i])

    # dalton
    theta_dalton[i] = bna_fit(key2, seirah_logpost_dalton, n_samples, phi_init, x0)
    theta_dalton[i] = np.exp(theta_dalton[i])

# # np.save("saves/seirah_theta_diffrax.npy", theta_diffrax)
# np.save("saves/seirah_theta_basic.npy", theta_basic)
# np.save("saves/seirah_theta_dalton.npy", theta_dalton)

# --- plot ---------------------------------------------------------------
theta_diffrax = np.load("saves/seirah_theta_diffrax.npy")
# theta_basic = np.load("saves/seirah_theta_basic.npy")
# theta_dalton = np.load("saves/seirah_theta_dalton.npy")
plot_theta = [theta_basic, theta_dalton]
theta_names = ["basic", "dalton"]
var_names = ["b", "r", r"$\alpha$", "$D_e$", "$D_I$", "$D_q$", "$E(0)}$", "$I(0)}$"]
param_true = np.append(theta, np.array([15492, 21752]))
clip = [(0.0, 5.0), None, (0.0, 1.5), None, None, None, None, None]
figure = theta_plot(plot_theta, theta_names, theta_diffrax, param_true, dt_obs/n_res_list, var_names, clip=clip, rows=2)
# figure.savefig('figures/seirahfigure.pdf')
