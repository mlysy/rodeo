r"""
This module implements the MCMC solver by Chkrebtii.
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from rodeo.ode import *
sim_jit = jax.jit(solve_sim, static_argnums=(1, 7, 11))
class oc_mcmc:

    def __init__(self, key, fun, W, tmin, tmax, phi_mean, phi_sd, y_obs,n_theta):
        self.key = key
        self.fun = fun
        self.W = W
        self.tmin = tmin
        self.tmax = tmax
        self.y_obs = y_obs
        # prior on parameters
        self.n_theta = n_theta
        self.phi_mean = phi_mean
        self.phi_sd = phi_sd
        # initialized after determining number of steps
        self.n_steps = None
        self.n_res = None
        self.prior_pars = None
        # Sigma for proposal
        self.Sigma_prop = None

    def logprior(self, phi):
        r"Calculate the loglikelihood of the normal distribution."
        return jnp.sum(jsp.stats.norm.logpdf(x=phi, loc=self.phi_mean, scale=self.phi_sd))

    def prop_lpdf(self, phi, phi_prime):
        r"Computes the proposal log pdf of theta."
        return jnp.sum(jsp.stats.multivariate_normal.logpdf(x=phi, mean=phi_prime, cov=self.Sigma_prop))
    
    def prop_sample(self, key, phi):
        r"Produce a draw of theta using proposal distribution."
        return jax.random.multivariate_normal(key=key, mean=phi, cov=self.Sigma_prop)

    def loglik(self, X_t):
        "Dependent on example"
        pass

    # def x0_initialize(self, phi, x0):
    #     r"Initialize x0 for none missing initial values"
    #     j = 0
    #     for ind in self.mask:
    #         x0 = x0.at[ind,0].set(phi[j])
    #         j+=1
    #     return x0

    def oc_solve(self, key, phi):
        r"Solve the ODE given the theta"
        x0 = jnp.expand_dims(phi[self.n_theta:],-1)
        theta = jnp.exp(phi[:self.n_theta])
        v0 = self.fun(x0, 0, theta)
        X_0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
        X_t = solve_sim(key, self.fun, self.W, X_0, theta, self.tmin, self.tmax, self.n_steps, **self.prior_pars, interrogate=interrogate_chkrebtii)
        # X_t = sim_jit(key, self.fun, self.W, X_0, theta, self.tmin, self.tmax, self.n_steps, **self.prior_pars, interrogate=interrogate_chkrebtii)
        X_t = X_t[::self.n_res, :, 0]
        return X_t

    def mcmc_sample(self, phi_init, n_samples):
        r"Compute MCMC samples via Chkrebtii method."
        
        def scan_fun(carry, t):
            phi_curr = carry['phi']
            X_curr = carry['X']
            key, *subkeys = jax.random.split(carry["key"], num=4)
            phi_prop = self.prop_sample(subkeys[0], phi_curr)
            X_prop = self.oc_solve(subkeys[1], phi_prop)
            lacc_prop = self.loglik(X_prop) + self.logprior(phi_prop) - self.prop_lpdf(phi_prop, phi_curr)
            lacc_curr = self.loglik(X_curr) + self.logprior(phi_curr) - self.prop_lpdf(phi_curr, phi_prop)
            mh_acc = jnp.exp(lacc_prop - lacc_curr)
            U = jax.random.uniform(subkeys[2])

            def _true_fun():
                return phi_prop, X_prop
            
            def _false_fun():
                return phi_curr, X_curr

            phi_next, X_next = jax.lax.cond(U<=mh_acc, _true_fun, _false_fun)
            # output
            carry = {
                "phi": phi_next,
                "X" : X_next,
                "key": key
            }
            stack = {
                "phi": phi_next
            }
            return carry, stack
        
        key, subkey = jax.random.split(self.key)
        X_init = self.oc_solve(subkey, phi_init)

        scan_init = {
            "phi" : phi_init,
            "X" : X_init,
            "key" : key
        }
        _, scan_out = jax.lax.scan(scan_fun, scan_init, jnp.arange(n_samples))
        return scan_out['phi']
