import jax
import jax.numpy as jnp
import jax.scipy as jsp

from rodeo.solve import *

class MarginalMCMC(object):

    def __init__(self,
                 obs_data, obs_times,
                 t_min, t_max, n_steps):
        r"""
        Base class for the marginal MCMC algorithm of Chkrebtii et al (2016).

        Args:
            obs_data (ndarray(n_obs, n_block, n_bobs)): Observation data.
            obs_times (ndarray(n_obs)): Time where observations are recorded.
            t_min (float): Starting time for ODE.
            t_max (float): Ending time for ODE.
            n_steps (int): Number of steps to evaluate the ODE.
        """
        self._obs_data = obs_data
        self._obs_times = obs_times
        self._t_min = t_min
        self._t_max = t_max
        self._n_steps = n_steps
        self._sim_times = jnp.linspace(t_min, t_max, n_steps + 1)
        self._obs_ind = jnp.searchsorted(self._sim_times, self._obs_times)

    def prop_lpdf(self, Theta_new, Theta_old, **params):
        r"""
        Computes the log-density of the proposal distribution ``q(Theta_new | Theta_old)``.

        The base class assumes that ``Theta`` is a JAX array and that proposal is ``Theta_new ~ind Normal(Theta_old, params["scale"]^2)``.

        Args:
            Theta_new (JAX pytree): New model parameters.
            Theta_old (JAX pytree): Old model parameters.
            params (optional): Parameters used in the proposal distribution.

        Return:
            (float): Log-density of proposal distribution.
        """
        lp = jsp.stats.norm.logpdf(
            x=Theta_new,
            loc=Theta_old,
            scale=params["scale"]
        )
        return jnp.sum(lp)

    def prop_sim(self, key, Theta_old, **params):
        r"""
        Simulate a draw from the proposal distribution.
        """
        eps = jax.random.normal(key, shape=Theta_old.shape)
        Theta_new = Theta_old + params["scale"] * eps
        return Theta_new

    def logprior(self, Theta):
        r"""
        The logprior for Theta. Dependent on example. Please implement this.

        Args:
            Theta: Parameters for the inference problem.
        """
        pass

    def obs_loglik(self, obs_data, ode_data, **params):
        r"""
        The loglikelihood of ``p(obs_data | ode_data, **params)``. Dependent on example. Please implement this.

        """
        pass

    def _logpost(self, ode_data, Theta, **params):
        "Calculate logposterior."
        ll = self.obs_loglik(self._obs_data, ode_data[self._obs_ind], **params)
        lpi = self.logprior(Theta)
        return ll + lpi

    def solve(self, key, Theta, t_min, t_max, n_steps):
        r"""
        Solve the ODE given :math:`\Theta` and return the indices where the observations exist.

        Args:
            key (PRNGKey): PRNG key.
            Theta (ndarray(n_theta)): Parameters for the inference problem.
            t_min (float): Starting time for ODE.
            t_max (float): Ending time for ODE.
            n_steps (int): Number of steps to evaluate the ODE.

        Return:
            (ndarray(n_block, n_bstate, n_bmeas)): Solution for given ``Theta`` using the Chkrebtii solver.
        """
        pass

    def initialize(self, key, Theta_init, **params):
        r"""
        Compute the initial loglikelihood of ``Theta_init``.

        Args:
            key (PRNGKey): PRNG key.
            Ttheta_init (ndarray(n_theta)): Initial parameters for the inference problem.

        Return:
            (tuple):
            - **Theta_init** (n_theta): Initial parameters for the inference problem.
            - **Xt_init** (ndarray(n_block, n_bstate, n_bmeas)): Solution for given ``Theta_init`` using the Chkrebtii solver.
            - **lp_init** (float): Loglikelihood of the ``Theta_init``.

        """
        Xt_init = self.solve(key, Theta_init, self._t_min, self._t_max, self._n_steps)
        lp_init = self._logpost(Xt_init, Theta_init, **params)
        return (Theta_init, Xt_init, lp_init)

    def step(self, key, state, **params):
        r"""
        Compute one step of the MCMC algorithm given the current state.

        Args:
            key (PRNGKey): PRNG key.
            state (tuple): Current state which constains the current parameter ``Theta``, ODE solution and loglikelihood.
            param (ndarrary(n_par)): Parameters used in the proposal distribution.

        Return:
            (tuple):
            - **state** (tuple): Next state.
            - **sample** (tuple): A sample of ``Theta`` using the MCMC algorithm.
        """
        keys = jax.random.split(key, num=3)
        Theta_curr, Xt_curr, lp_curr = state
        Theta_prop = self.prop_sim(keys[0], Theta_curr, **params)
        _, Xt_prop, lp_prop = self.initialize(keys[1], Theta_prop, **params)
        lacc_prop = lp_prop - self.prop_lpdf(Theta_prop, Theta_curr, **params)
        lacc_curr = lp_curr - self.prop_lpdf(Theta_curr, Theta_prop, **params)
        mh_acc = jnp.exp(lacc_prop - lacc_curr)
        U = jax.random.uniform(keys[2])

        def _true_fun():
            return Theta_prop, Xt_prop, lp_prop

        def _false_fun():
            return state

        accept = U <= mh_acc
        state = jax.lax.cond(accept, _true_fun, _false_fun)

        sample = {
            "Theta": state[0],
            "accept": accept
        }
        return state, sample
