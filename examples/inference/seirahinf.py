from scipy.integrate import odeint
from .inference import inference
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from rodeo.ode import *

class seirahinf(inference):
    r"Inference using the France Covid data from Prague et al."
    
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

    def loglike(self, Y_t, X_t, step_size, obs_size, theta):
        data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.thinning(ode_tseq, data_tseq, X_t)
        X_in = self.covid_obs(X_t, theta)
        return jnp.sum(jsp.stats.poisson.logpmf(Y_t, X_in))

    def x0_initialize(self, phi, x0, phi_len):
        j = 0
        xx0 = []
        for i in range(len(x0)):
            if x0[i] is None:
                xx0.append(jnp.exp(phi[phi_len+j]))
                j+=1
            else:
                xx0.append(x0[i])
        return jnp.array(xx0)
    
    def covid_obs(self, X_t, theta):
        r"Compute the observations as detailed in the paper"
        I_in = theta[1]*X_t[:,1]/theta[3]
        H_in = X_t[:,2]/theta[5]
        X_in = jnp.array([I_in, H_in]).T
        return X_in
    
    def simulate(self, x0, theta, tseq):
        r"""Get the observations for the SEIRAH Covid example.
        None of the compartments are directly observed, however 
        the daily infections and hospitalizations are observed. 
        They can be computed as
        
        .. math::

            I^{(in)}(t) = rE(t)/D_e
            H^{(in)}(t) = I(t)/D_q

        """
        X_t = odeint(self.ode_fun, x0, tseq, args=(theta,))
        X_in = self.covid_obs(X_t, theta)
        Y_in = np.random.default_rng(111).poisson(X_in)
        return Y_in, X_in
    