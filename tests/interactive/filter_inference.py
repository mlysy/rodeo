from math import ceil
from functools import partial
import numpy as np
import scipy as sp
import scipy.stats
import jax
from jax import random, jacfwd, jacrev, grad, lax
import jax.numpy as jnp
import jax.scipy as jsp
from double_filter import *
from fenrir_filter import *

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import warnings
warnings.filterwarnings('ignore')

# jit double filter and fenrir
df_jit = jax.jit(double_ode_filter, static_argnums=(1, 14, 15))
f_jit = jax.jit(fenrir_filter, static_argnums=(0, 5))
ng_jit = jax.jit(fenrir_filterng, static_argnums=(0, 5, 13, 14))

class filter_inference:
    r"""
    Perform parameter inference for the model via mode/quadrature using Euler's 
    approximation and the KalmanODE solver. 

    Args:
        key (PRNGKey): PRNG key.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking 
            arguments :math:`x` and :math:`t`.
        data_tseq (ndarray(n)): Time points of the observed data.
        ode_tseq (ndarray(N)): Time points of the ODE solution.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`x_t` at 
            time :math:`t = 0`.
        theta_true (ndarray(n_theta)): True value of :math:`\theta` in the ODE function.
        gamma (float): Noise parameter to simulate the observations.
        phi (ndarray(n_theta)): Log of observed :math:`\theta`.
        Y_t (ndarray(n_steps, n_state)): Simulated observations.
        step_size (float): Distance between discretisation points.
        phi_mean (ndarray(n_theta)): Mean of :math:`\phi`.
        phi_sd (ndarray(n_theta)): Standard deviation of :math:`\phi`.
        kalman_solve (fun): Kalman solve method defined by the parameter inference problem.
        euler_solve (fun): Euler solve method defined by the parameter inference problem.
        solve (fun): Either kalman_solve or euler_solve.
        theta (ndarray(n_theta)): Observed :math:`\theta`.
        phi_hat (ndarray(n_theta)): Optimized observed :math:`\phi`.
        phi_var (ndarray(n_theta, n_theta)): Variance matrix of phi_hat.
        n_samples (int): Number of samples of :math:`\theta` to simulate.
        theta_euler (ndarray(n_samples, n_theta)): Simulated n_samples of 
            :math:`\theta` using Euler's approximation.
        theta_kalman (ndarray(n_samples, n_theta)): Simulated n_samples of 
            :math:`\theta` using KalmanODE solver.
    """
    def __init__(self, key, tmin, tmax, fun):
        self.key = key
        self.tmin = tmin
        self.tmax = tmax
        self.fun = fun
        self.n_res = None
        self.W = None
        self.mu_state = None
        self.wgt_state = None
        self.var_state = None
        self.mu_obs = None
        self.wgt_obs = None
        self.var_obs = None
        self.gamma = None
        self.funpad = None
        self.x_fun = None

    def logprior(self, x, mean, sd):
        r"Calculate the loglikelihood of the lognormal distribution."
        return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))
    
    def x0_initialize(self, phi, x0, phi_len):
        j = 0
        xx0 = []
        for i in range(len(x0)):
            if x0[i] is None:
                xx0.append(phi[phi_len+j])
                j+=1
            else:
                xx0.append(x0[i])
        return jnp.array(xx0)

    def double_filter_nlpost(self, phi, Y_t, x0, phi_mean, phi_sd, double, varzero):
        r"Compute the negative loglikihood of :math:`Y_t` using the double filter algorithm."
        phi_ind = len(phi_mean)
        xx0 = self.x0_initialize(phi, x0, phi_ind)
        phi = phi[:phi_ind]
        theta = jnp.exp(phi)
        xx0 = self.funpad(xx0, 0, theta)
        self.key, subkey = jax.random.split(self.key)
        lp = df_jit(subkey, self.fun, xx0, theta, self.tmin, self.tmax, self.W, 
                    self.wgt_state, self.mu_state, self.var_state,
                    self.wgt_obs, self.mu_obs, self.var_obs, Y_t, double, varzero)
        lp += self.logprior(phi, phi_mean, phi_sd)
        return -lp

    def fenrir_nlpost(self, phi, Y_t, x0, phi_mean, phi_sd, double, varzero):
        r"Compute the negative loglikihood of :math:`Y_t` using the fenrir algorithm."
        phi_ind = len(phi_mean)
        xx0 = self.x0_initialize(phi, x0, phi_ind)
        phi = phi[:phi_ind]
        theta = jnp.exp(phi)
        xx0 = self.funpad(xx0, 0, theta)
        lp = f_jit(self.fun, xx0, theta, self.tmin, self.tmax, self.n_res, self.W, 
                   self.wgt_state, self.mu_state, self.var_state,
                   self.wgt_obs, self.mu_obs, self.var_obs, Y_t)
        lp += self.logprior(phi, phi_mean, phi_sd)
        return -lp

    def fenrir_nlpostng(self, phi, Y_t, x0, phi_mean, phi_sd, double, varzero):
        r"Compute the negative loglikihood of :math:`Y_t` using the non-gaussian algorithm."
        phi_ind = len(phi_mean)
        xx0 = self.x0_initialize(phi, x0, phi_ind)
        phi = phi[:phi_ind]
        theta = jnp.exp(phi)
        xx0 = self.funpad(xx0, 0, theta)
        lp = ng_jit(self.fun, xx0, theta, self.tmin, self.tmax, self.n_res, self.W, 
                    self.wgt_state, self.mu_state, self.var_state,
                    self.wgt_obs, Y_t, self.gamma, self.obs_fun, self.x_fun)
        lp += self.logprior(phi, phi_mean, phi_sd)
        return -lp

    def phi_fit(self, Y_t, x0, phi_mean, phi_sd, phi_init, obj_fun, method="Newton-CG", double=True, varzero=True):
        r"""Compute the optimized :math:`\log{\theta}` and its variance given 
            :math:`Y_t`."""
        
        n_phi = len(phi_init)
        gradf = grad(obj_fun)
        hes = jacfwd(jacrev(obj_fun))
        opt_res = sp.optimize.minimize(obj_fun, phi_init,
                                       args=(Y_t, x0, phi_mean, phi_sd, double, varzero),
                                       method=method,
                                       jac=gradf)
        phi_hat = opt_res.x
        phi_fisher = hes(phi_hat, Y_t, x0, phi_mean, phi_sd, double, varzero)
        # phi_cho, low = jsp.linalg.cho_factor(phi_fisher)
        # phi_var = jsp.linalg.cho_solve((phi_cho, low), jnp.eye(n_phi))
        phi_var = jsp.linalg.inv(phi_fisher)
        return phi_hat, phi_var, phi_fisher

        
    def phi_sample(self, phi_hat, phi_var, n_samples):
        r"""Simulate :math:`\theta` given the :math:`\log{\hat{\theta}}` 
            and its variance."""
        phi = np.random.default_rng(12345).multivariate_normal(phi_hat, phi_var, n_samples)
        return phi

    def theta_plotsingle(self, theta, theta_diffrax, theta_true, res_sizes, var_names, clip=None, rows=1):
        r"""Plot the distribution of :math:`\theta` using the Kalman solver 
            and the Euler approximation."""
        n_lst, _, n_theta = theta.shape
        ncol = ceil(n_theta/rows) +1
        fig = plt.figure(figsize=(20, 5*rows))
        patches = [None]*(n_lst+2)
        if clip is None:
            clip = [None]*ncol*rows 
        carry = 0
        for t in range(1,n_theta+1):
            row = (t-1)//(ncol-1)
            if t%(ncol)==0:
                carry +=1
            
            axs = fig.add_subplot(rows, ncol, t+carry)
            axs.set_title(var_names[t-1])
            axs.axvline(x=theta_true[t-1], linewidth=1, color='r', linestyle='dashed')
            axs.set_yticks([])

            for h in range(n_lst):
                if t==1:
                    patches[h] = mpatches.Patch(color='C{}'.format(h), label='n_res ={}'.format(res_sizes[h]))
                sns.kdeplot(theta[h, :, t-1], ax=axs, clip=clip[t-1])
            
            sns.kdeplot(theta_diffrax[:, t-1], ax=axs, color='black', clip=clip[t-1])
            
            if t==n_theta:
                patches[-2] = mpatches.Patch(color='black', label="True Posterior")
                patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\Theta$')
                
        fig.legend(handles=patches, framealpha=0.5, loc=7)
        
        fig.tight_layout()
        plt.show()
        return fig
