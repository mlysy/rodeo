import jax
import jax.numpy as jnp
import jax.scipy as jsp

from rodeo.ode import *

class oc_mcmc:
    r"""
    This module implements the MCMC solver by Chkrebtii et al 2016. 
    """
    def __init__(self, fun, W, x0, tmin, tmax, n_steps, n_res, prior_pars, y_obs):
        """
        Base class for the MCMC solver by Chkrebtii. The main method is the :meth:`oc_mcmc.oc_mcmc.step` which
        returns one sample of theta.
        
        Args:
            key (PRNGKey): PRNG key.
            fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
            W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
            x0 (ndarray(n_block, n_bstate)): Initial value of the state variable. 
            tmin (float): First time point of the time interval to be evaluated; :math:`a`.
            tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
            n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated.
            n_res (int): Resolution number.
            prior_pars (dict): Dictionary containing the IBM prior parameters required for rodeo solver.
            y_obs (ndarray(n_block, n_bmeas)): Noisy observations.
        
        """
        self.fun = fun
        self.W = W
        self.x0 = x0
        self.tmin = tmin
        self.tmax = tmax
        self.n_steps = n_steps
        self.n_res = n_res
        self.prior_pars = prior_pars
        self.y_obs = y_obs
        self.param = None

    def prop_lpdf(self, theta, theta_prime, param):
        r"""
        Computes the proposal log-PDF of :math:`\theta`. The base class assumes the proposal is N(theta; theta_prime, param).
        
        Args:
            theta (ndarray(n_theta)): Parameters for the inference problem.
            theta_prime (ndarray(n_theta)): :math:`\theta'` used in Chkrebtii MCMC method.
            param (ndarrary(n_par)): Parameters used in the proposal distribution.
        
        Return:
            (float): Loglikelihood of :math:`\theta`.
        """
        return jnp.sum(jsp.stats.multivariate_normal.logpdf(x=theta, mean=theta_prime, cov=param))
    
    def prop_sample(self, key, theta, param):
        r"""
        Produce a draw of theta using proposal distribution. The base assumes the proposal is N(theta, param).

        Args:
            theta (ndarray(n_theta)): Parameters for the inference problem.
            param (ndarrary(n_par)): Parameters used in the proposal distribution.
        
        Return:
            (ndarray(n_theta)): Sample of :math:`\theta`.
        """
        return jax.random.multivariate_normal(key=key, mean=theta, cov=param)

    def logprior(self, theta):
        r"""
        The logprior for theta. Dependent on example. Please implement this.

        Args:
            theta (ndarray(n_theta)): Parameters for the inference problem.

        """
        pass

    def loglik(self, X_t):
        r"""
        The loglikelihood of p(Y_t | X_t). Dependent on example. Please implement this.
        
        Args:
            X_t (ndarray(n_block, n_bstate, n_bmeas)): Solution for given initial :math:`\theta` using the Chkrebtii solver.
        
        """
        pass

    def solve(self, key, theta):
        r"""
        Solve the ODE given :math:`\theta` and return the indices where the observations exist.
        
        Args:
            key (PRNGKey): PRNG key.
            theta (ndarray(n_theta)): Parameters for the inference problem.
        
        Return:
            (ndarray(n_block, n_bstate, n_bmeas)): Solution for given :math:`\theta` using the Chkrebtii solver.
        """
        X_t = solve_sim(key, self.fun, self.W, self.x0, theta, self.tmin, self.tmax, self.n_steps, **self.prior_pars, interrogate=interrogate_chkrebtii)
        X_t = X_t[::self.n_res, :, 0]
        return X_t

    def init(self, key, theta_init):
        r"""
        Compute the initial loglikelihood of theta_init and X_init.

        Args:
            key (PRNGKey): PRNG key.
            theta_init (ndarray(n_theta)): Initial parameters for the inference problem.
        
        Return:
            (dict):
            - **theta** (n_theta): Initial parameters for the inference problem.
            - **X_t** (ndarray(n_block, n_bstate, n_bmeas)): Solution for given initial :math:`\theta` using the Chkrebtii solver.
            - **ll** (float): Loglikelihood of the initial :math:`\theta`.
        
        """
        X_init = self.solve(key, theta_init)
        ll_init = self.loglik(X_init) + self.logprior(theta_init)
        return {
            "theta": theta_init,
            "X_t": X_init,
            "ll": ll_init,
            "acc": 0,
            "n_sam": 0
        }

    def step(self, key, state, param):
        r"""
        Compute one step of the MCMC algorithm given the current state.

        Args:
            key (PRNGKey): PRNG key.
            state (dict): Current state which constains the current parameter :math:`\theta`, ODE solution for :math:`\theta` and loglikelihood.
            param (ndarrary(n_par)): Parameters used in the proposal distribution.
        
        Return:
            (tuple):
            - **state** (dict): Next state.
            - **sample** (dict): A sample of :math:`\theta` using the MCMC algorithm.
        """
        keys = jax.random.split(key, num=3)
        theta_prev = state['theta']
        X_prev = state['X_t']
        ll_prev = state['ll']
        n_sam = state['n_sam']
        acc = state['acc']
        theta_prop = self.prop_sample(keys[0], theta_prev, self.param)
        X_prop = self.solve(keys[1], theta_prop)
        ll_prop = self.loglik(X_prop) + self.logprior(theta_prop)
        lacc_prop = ll_prop - self.prop_lpdf(theta_prop, theta_prev, self.param)
        lacc_prev = ll_prev - self.prop_lpdf(theta_prev, theta_prop, self.param)
        mh_acc = jnp.exp(lacc_prop - lacc_prev)
        U = jax.random.uniform(keys[2])

        def _true_fun():
            return theta_prop, X_prop, ll_prop, 1
            
        def _false_fun():
            return theta_prev, X_prev, ll_prev, 0

        theta_curr, X_curr, ll_curr, ar = jax.lax.cond(U<=mh_acc, _true_fun, _false_fun)
        
        state = {
            "theta": theta_curr,
            "X_t": X_curr,
            "ll": ll_curr,
            "acc": acc + ar,
            "n_sam": n_sam + 1
        }
        sample = {
            "theta": theta_curr
        }
        return state, sample
