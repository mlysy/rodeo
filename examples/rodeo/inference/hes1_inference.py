import jax.numpy as jnp
import numpy as np

from scipy.integrate import odeint
from .inference import inference
from diffrax import SaveAt
from rodeo.ode import *

class hes1_inference(inference):
    r"Inference for the Hes1 model"
    def __init__(self, key, fun, W, tmin, tmax, phi_mean, phi_sd, mask, noise_sigma, n_theta):
        super().__init__(key, fun, W, tmin, tmax, phi_mean, phi_sd, mask, None)
        self.n_theta = n_theta
        self.noise_sigma = noise_sigma
    
    def ode_fun(self, X_t, t, theta):
        "Hes1 ODE written for odeint"
        P, M, H = jnp.exp(X_t)
        a, b, c, d, e, f, g = theta
        x1 = -a*H + b*M/P - c
        x2 = -d + e/(1+P*P)/M
        x3 = -a*P + f/(1+P*P)/H - g
        return jnp.array([x1, x2, x3])

    def rax_fun(self, t, X_t, theta):
        "Hes1 ODE written for diffrax"
        P, M, H = jnp.exp(X_t)
        a, b, c, d, e, f, g = theta
        x1 = -a*H + b*M/P - c
        x2 = -d + e/(1+P*P)/M
        x3 = -a*P + f/(1+P*P)/H - g
        return jnp.array([x1, x2, x3])

    def loglike(self, Y_t, X_t, theta):
        r"loglikelihood function for the Hes1 noise model"
        X_t = self.hes1_obs(X_t)
        return self.logprior(Y_t, X_t, self.noise_sigma)

    def hes1_obs(self, sol):
        r"Given the solution process, get the corresponding observations"
        return jnp.append(sol[::2, 0], sol[1::2, 1])

    def simulate(self, x0, theta):
        r"Get the observations assuming a normal distribution."
        tseq = jnp.linspace(self.tmin, self.tmax, 33)
        # sol = odeint(self.ode_fun, x0, tseq, args=(theta,))
        X_t = solve_mv(self.key, self.fun, self.W, x0, theta, self.tmin, self.tmax, self.n_steps, **self.prior_pars)[0]
        sol = X_t[::self.n_res, :, 0]
        X_t = self.hes1_obs(sol)
        e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + self.noise_sigma*e_t
        self.y_obs = Y_t
        self.saveat = SaveAt(ts = tseq)
        return Y_t, X_t
