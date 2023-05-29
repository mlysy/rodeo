import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from scipy.integrate import odeint
from .inference import inference
from diffrax import SaveAt
from rodeo.ode import *

class seirah_inference(inference):
    r"Inference using the France Covid data from Prague et al."
    def __init__(self, key, fun, W, tmin, tmax, phi_mean, phi_sd, mask, n_theta):
        super().__init__(key, fun, W, tmin, tmax, phi_mean, phi_sd, mask, None)
        self.n_theta = n_theta
        
    def ode_fun(self, X_t, t, theta):
        "SEIRAH ODE function"
        p = len(X_t)//6
        S, E, I, R, A, H = X_t[::p]
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

    def rax_fun(self, t, X_t, theta):
        "SEIRAH ODE function"
        p = len(X_t)//6
        S, E, I, R, A, H = X_t[::p]
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

    def x0_initialize(self, phi, x0):
        r"Initialize x0 for none missing initial values"
        j = 0
        for ind in self.mask:
            x0 = x0.at[ind,0].set(jnp.exp(phi[j]))
            j+=1
        return x0

    def loglike(self, Y_t, X_t, theta):
        r"loglikelihood function for the SEIRAH noise model"
        X_in = self.covid_obs(X_t, theta)
        return jnp.sum(jsp.stats.poisson.logpmf(Y_t, X_in))

    def covid_obs(self, X_t, theta):
        r"Compute the observations as detailed in the paper"
        I_in = theta[1]*X_t[:,1]/theta[3]
        H_in = X_t[:,2]/theta[5]
        X_in = jnp.array([I_in, H_in]).T
        return X_in
    
    def simulate(self, x0, theta):
        r"""Get the observations for the SEIRAH Covid example.
        None of the compartments are directly observed, however 
        the daily infections and hospitalizations are observed. 
        They can be computed as
        
        .. math::

            I^{(in)}(t) = rE(t)/D_e
            H^{(in)}(t) = I(t)/D_q

        """
        tseq = jnp.linspace(self.tmin, self.tmax, 61)
        # X_t = odeint(self.ode_fun, x0, tseq, args=(theta,))
        X_t = solve_mv(self.key, self.fun, self.W, x0, theta, self.tmin, self.tmax, self.n_steps, **self.prior_pars)[0]
        X_t = X_t[::self.n_res, :, 0]
        X_in = self.covid_obs(X_t, theta)
        Y_in = np.random.default_rng(111).poisson(X_in)
        self.y_obs = Y_in
        self.saveat = SaveAt(ts = tseq)
        return Y_in, X_in
    