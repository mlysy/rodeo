from scipy.integrate import odeint
from .inference import inference
import numpy as np
import jax
from jax.config import config
config.update("jax_enable_x64", True)

from rodeo.ode_solve import *

class hes1inf(inference):
    r"Inference for the Hes1 model"
    
    def ode_fun(self, X_t, t, theta):
        "Hes1 ODE written for odeint"
        P, M, H = jnp.exp(X_t)
        a, b, c, d, e, f, g = theta
        x1 = -a*H + b*M/P - c
        x2 = -d + e/(1+P*P)/M
        x3 = -a*P + f/(1+P*P)/H - g
        return jnp.array([x1, x2, x3])

    def rax_fun(self, t, X_t, theta):
        "Hes1 ODE written for odeint"
        P, M, H = jnp.exp(X_t)
        a, b, c, d, e, f, g = theta
        x1 = -a*H + b*M/P - c
        x2 = -d + e/(1+P*P)/M
        x3 = -a*P + f/(1+P*P)/H - g
        return jnp.array([x1, x2, x3])

    def loglike(self, Y_t, X_t, step_size, obs_size, theta, gamma):
        data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.thinning(ode_tseq, data_tseq, X_t)
        X_t = self.hes1_obs(X_t)
        return self.logprior(Y_t, X_t, gamma)

    def hes1_obs(self, sol):
        r"Given the solution process, get the corresponding observations"
        return jnp.append(sol[::2, 0], sol[1::2, 1])

    def simulate(self, x0, theta, gamma, tseq):
        r"Get the observations assuming a normal distribution."
        sol = odeint(self.ode_fun, x0, tseq, args=(theta,))
        X_t = self.hes1_obs(sol)
        e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t, X_t
    