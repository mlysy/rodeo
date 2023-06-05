import jax
import jax.numpy as jnp
import jax.scipy as jsp

from rodeo.ode import *
from rodeo.oc_mcmc import oc_mcmc

class fitz_ocmcmc(oc_mcmc):

    def __init__(self, fun, W, tmin, tmax, phi_mean, phi_sd, y_obs, n_theta, noise_sigma):
        super().__init__(fun, W, None, tmin, tmax, None, None, None, y_obs)
        self.phi_mean = phi_mean
        self.phi_sd = phi_sd
        self.n_theta = n_theta
        self.noise_sigma = noise_sigma

    def logprior(self, phi):
        r"Calculate the loglikelihood of the prior."
        return jnp.sum(jsp.stats.norm.logpdf(x=phi, loc=self.phi_mean, scale=self.phi_sd))

    def loglik(self, X_t):
        r"Calculate the loglikelihood of the observations."
        return jnp.sum(jsp.stats.norm.logpdf(x=self.y_obs, loc=X_t, scale=self.noise_sigma))

    def solve(self, key, phi):
        r"Solve the ODE given the theta"
        x0 = jnp.expand_dims(phi[self.n_theta:],-1)
        theta = jnp.exp(phi[:self.n_theta])
        v0 = self.fun(x0, 0, theta)
        X_0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
        X_t = solve_sim(key, self.fun, self.W, X_0, theta, self.tmin, self.tmax, self.n_steps, **self.prior_pars, interrogate=interrogate_chkrebtii)
        X_t = X_t[::self.n_res, :, 0]
        return X_t
    
    def mcmc_sample(self, key, phi_init, n_samples):
        param = jnp.diag(jnp.array([0.0001, 0.01, 0.0001, 0.0001, 0.0001]))
        key, subkey = jax.random.split(key)
        initial_state = self.init(subkey, phi_init)
        def one_step(state, key):
            state, sample = self.step(key, state, param)
            return state, sample

        keys = jax.jax.random.split(key, n_samples)
        _, samples = jax.lax.scan(one_step, initial_state, keys)
        return samples['theta']
