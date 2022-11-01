from scipy.integrate import odeint
from .inference import inference
import numpy as np
import jax

from rodeo.ode_solve import *
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

class fitzinf(inference):
    r"Inference assuming a normal prior"
    
    def ode_fun(self, X_t, t, theta):
        "Fitz ODE written for odeint"
        a, b, c = theta
        p = len(X_t)//2
        V, R = X_t[0], X_t[p]
        return jnp.array([c*(V - V*V*V/3 + R),
                        -1/c*(V - a + b*R)])

    def rax_fun(self, t, X_t, theta):
        "Fitz ODE written for diffrax"
        a, b, c = theta
        p = len(X_t)//2
        V, R = X_t[0], X_t[p]
        return jnp.array([c*(V - V*V*V/3 + R),
                        -1/c*(V - a + b*R)])

    def loglike(self, Y_t, X_t, step_size, obs_size, theta, gamma):
        data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.thinning(ode_tseq, data_tseq, X_t)
        return self.logprior(Y_t, X_t, gamma)

    def simulate(self, x0, theta, gamma, tseq):
        r"Get the observations assuming a normal distribution."
        X_t = odeint(self.ode_fun, x0, tseq, args=(theta,))
        e_t = np.random.default_rng(100).normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t, X_t
