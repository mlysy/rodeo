import jax
import jax.numpy as jnp
import jax.scipy as jsp

from rodeo.ode import *
from .oc_mcmc import oc_mcmc

class fitz_ocmcmc(oc_mcmc):

    def __init__(self, key, fun, W, tmin, tmax, phi_mean, phi_sd, y_obs, n_theta, noise_sigma):
        super().__init__(key, fun, W, tmin, tmax, phi_mean, phi_sd, y_obs, n_theta)
        self.noise_sigma = noise_sigma
        self.Sigma_prop = jnp.diag(jnp.array([0.0001, 0.01, 0.0001, 0.0001, 0.0001]))

    def loglik(self, X_t):
        return jnp.sum(jsp.stats.norm.logpdf(x=self.y_obs, loc=X_t, scale=self.noise_sigma))

