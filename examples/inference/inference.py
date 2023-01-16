import numpy as np
import jax
from jaxopt import ScipyMinimize
from jax import jacfwd, jacrev
import jax.numpy as jnp
import jax.scipy as jsp

from rodeo.ode import *
from euler import euler

import warnings
from jax.config import config
warnings.filterwarnings('ignore')
config.update("jax_enable_x64", True)

from diffrax import diffeqsolve, Dopri5, ODETerm, PIDController

# mv_jit = jax.jit(solve_mv, static_argnums=(1, 7))

class inference:
    r"""
    Perform parameter inference for the model via mode/quadrature using the rodeo solver. 

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
        y_obs (ndarray(n_steps, n_state)): Simulated observations :math:`Y_t`.
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
        theta_rodeo (ndarray(n_samples, n_theta) ): Simulated n_samples of 
            :math:`\theta` using rodeo solver.
    """
    def __init__(self, key, fun, W, tmin, tmax, phi_mean, phi_sd, mask, y_obs=None):
        self.key = key
        self.fun = fun
        self.W = W
        self.tmin = tmin
        self.tmax = tmax
        self.y_obs = y_obs
        # initial x0 using mask
        self.mask = mask
        # prior on parameters
        self.phi_mean = phi_mean
        self.phi_sd = phi_sd
        # initialized after determining number of steps
        self.n_steps = None
        self.n_res = None
        self.prior_pars = None
        # diffrax
        self.term = ODETerm(self.rax_fun)
        self.solver = Dopri5()
        self.saveat = None
        self.diff_dt0 = None
        self.stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
        
    def logprior(self, x, mean, sd):
        r"Calculate the loglikelihood of the normal distribution."
        return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

    def x0_initialize(self, phi, x0):
        r"Initialize x0 for none missing initial values"
        j = 0
        for ind in self.mask:
            x0 = x0.at[ind,0].set(phi[j])
            j+=1
        return x0
    
    def kalman_nlpost(self, phi, x0):
        r"Compute the negative loglikihood of :math:`Y_t` using the KalmanODE."
        n_phi = len(self.phi_mean)
        x0 = self.x0_initialize(phi[self.n_theta:], x0)
        # print(x0)
        theta = jnp.exp(phi[:self.n_theta])
        v0 = self.fun(x0, 0, theta)
        X_0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
        X_t = solve_mv(self.key, self.fun, self.W, X_0, theta, self.tmin, self.tmax, self.n_steps, **self.prior_pars)[0]
        X_t = X_t[::self.n_res, :, 0]
        lp = self.loglike(self.y_obs, X_t, theta)
        lp += self.logprior(phi[:n_phi], self.phi_mean, self.phi_sd)
        return -lp
    
    def euler_nlpost(self, phi, x0):
        r"Compute the negative loglikihood of :math:`Y_t` using the Euler method."
        n_phi = len(self.phi_mean)
        x0 = self.x0_initialize(phi[self.n_theta:], x0)
        x0 = x0.flatten()
        theta = jnp.exp(phi[:self.n_theta])
        X_t = euler(self.ode_fun, x0, theta, self.tmin, self.tmax, self.n_steps)
        X_t = X_t[::self.n_res]
        lp = self.loglike(self.y_obs, X_t, theta)
        lp += self.logprior(phi[:n_phi], self.phi_mean, self.phi_sd)
        return -lp
    
    def diffrax_nlpost(self, phi, x0):
        r"Compute the negative loglikihood of :math:`Y_t` using a deterministic solver."
        n_phi = len(self.phi_mean)
        x0 = self.x0_initialize(phi[self.n_theta:], x0)
        x0 = x0.flatten()
        theta = jnp.exp(phi[:self.n_theta])
        X_t = diffeqsolve(self.term, self.solver, args = theta, t0=self.tmin, t1=self.tmax, dt0 = self.diff_dt0, 
                          y0=jnp.array(x0), saveat=self.saveat, stepsize_controller=self.stepsize_controller).ys
        lp = self.loglike(self.y_obs, X_t, theta)
        lp += self.logprior(phi[:n_phi], self.phi_mean, self.phi_sd)
        return -lp

    def phi_fit(self, phi_init, x0, obj_fun):
        r"""Compute the optimized :math:`\log{\theta}` and its variance given 
            :math:`Y_t`."""
        
        n_phi = len(phi_init)
        # obj_fun = jax.jit(obj_fun)
        hes = jacfwd(jacrev(obj_fun))
        solver = ScipyMinimize(method="Newton-CG", fun = obj_fun)
        opt_res = solver.run(phi_init, x0)
        phi_hat = opt_res.params
        phi_fisher = hes(phi_hat, x0)
        phi_cho, low = jsp.linalg.cho_factor(phi_fisher)
        phi_var = jsp.linalg.cho_solve((phi_cho, low), jnp.eye(n_phi))
        return phi_hat, phi_var
        
    def phi_sample(self, phi_hat, phi_var, n_samples):
        r"""Simulate :math:`\theta` given the :math:`\log{\hat{\theta}}` 
            and its variance."""
        phi = np.random.default_rng(12345).multivariate_normal(phi_hat, phi_var, n_samples)
        return phi
    
    
