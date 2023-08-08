import jax.numpy as jnp
import numpy as np

from scipy.integrate import odeint
from .inference import inference
from diffrax import SaveAt
from rodeo.ode import *

class fitz_inference(inference):
    r"Inference assuming a normal prior"
    def __init__(self, key, fun, W, tmin, tmax, phi_mean, phi_sd, mask, noise_sigma, n_theta):
        super().__init__(key, fun, W, tmin, tmax, phi_mean, phi_sd, mask, None)
        self.n_theta = n_theta
        self.noise_sigma = noise_sigma
        self.mean_obs = jnp.zeros((2, 1))
        self.trans_obs = jnp.array([[[1., 0., 0.]], [[1., 0., 0.]]])
        self.var_obs = noise_sigma**2*jnp.array([[[1.]],[[1.]]])

    def ode_fun(self, X_t, t, theta):
        "Fitz ODE written for odeint"
        a, b, c = theta
        p = len(X_t)//2
        V, R = X_t
        return jnp.hstack([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

    def rax_fun(self, t, X_t, theta):
        "Fitz ODE written for diffrax"
        a, b, c = theta
        p = len(X_t)//2
        V, R = X_t[0], X_t[p]
        return jnp.array([c*(V - V*V*V/3 + R),
                         -1/c*(V - a + b*R)])

    def loglike(self, Y_t, X_t, theta):
        r"loglikelihood function for the FitzHugh noise model"
        return self.logprior(Y_t, X_t, self.noise_sigma)

    def simulate(self, x0, theta):
        r"Get the observations assuming a normal distribution."
        tseq = jnp.linspace(self.tmin, self.tmax, 41)
        # X_t = odeint(self.ode_fun, x0, tseq, args=(theta,))
        X_t = solve_mv(self.key, self.fun, self.W, x0, theta, self.tmin, self.tmax, self.n_steps, **self.prior_pars)[0]
        X_t = X_t[::self.n_res, :, 0]
        e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + self.noise_sigma*e_t
        self.y_obs = Y_t
        self.saveat = SaveAt(ts = tseq)
        return Y_t, X_t
