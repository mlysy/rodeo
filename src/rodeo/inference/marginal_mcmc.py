import jax
import jax.numpy as jnp
import jax.scipy as jsp
import rodeo
import rodeo.interrogate
from abc import abstractmethod


class MarginalMCMC(object):

    def __init__(self,
                 ode_fun,
                 obs_data, obs_times,
                 t_min, t_max, n_steps):
        r"""
        Base class for the marginal MCMC algorithm of Chkrebtii et al (2016).

        Args:
            ode_fun (function): ODE function.
            obs_data (ndarray(n_obs, n_block, n_bobs)): Observation data.
            obs_times (ndarray(n_obs)): Time where observations are recorded.
            t_min (float): Starting time for ODE.
            t_max (float): Ending time for ODE.
            n_steps (int): Number of steps to evaluate the ODE.
        """
        self._ode_fun = ode_fun
        self._obs_data = obs_data
        self._obs_times = obs_times
        self._t_min = t_min
        self._t_max = t_max
        self._n_steps = n_steps
        self._sim_times = jnp.linspace(t_min, t_max, n_steps + 1)
        self._obs_ind = jnp.searchsorted(self._sim_times, self._obs_times)

    def prop_lpdf(self, upars_prop, upars_curr, **prop_params):
        r"""
        Computes the log-density of the proposal distribution ``q(upars_prop | upars_curr)``.

        The base class assumes that ``upars_prop`` is a JAX array and that proposal is ``upars_prop ~ind Normal(upars_curr, prop_params["scale"]^2)``.

        Args:
            upars_prop (JAX pytree): Proposal model parameters.
            upars_curr (JAX pytree): Current model parameters.
            prop_params (optional): Parameters used in the proposal distribution.

        Return:
            (float): Log-density of proposal distribution.
        """
        lp = jsp.stats.norm.logpdf(
            x=upars_prop,
            loc=upars_curr,
            scale=prop_params["scale"]
        )
        return jnp.sum(lp)

    def prop_sim(self, key, upars_curr, **prop_params):
        r"""
        Simulate a draw from the proposal distribution.
        """
        eps = jax.random.normal(key, shape=upars_curr.shape)
        upars_prop = upars_curr + prop_params["scale"] * eps
        return upars_prop

    @abstractmethod
    def parse_pars(self, upars, dt):
        r"""
        Parse the parameters from ``upars`` such that the output contains a tuple which contains the elements:
            
            1. The weight matrix defining the ODE; :math:`W`.
            2. The initial value to the IVP (i.e., ``ode_init``).
            3. The Gaussian Prior parameters (i.e., ``prior_weight``, ``prior_var``).
            4. A dictionary containing the other parameters (i.e., model parameter :math:`\theta`, measurement parameter :math:`\phi`)
        
        The order should be ``(ode_init, prior_weight, prior_var, dictionary)``.
        Please implement this.
        """
        pass

    @abstractmethod
    def logprior(self, upars):
        r"""
        The logprior for ``upars``. Dependent on example. Please implement this.

        Args:
            upars: All the parameters in the inference problem.
        """
        pass

    @abstractmethod
    def obs_loglik(self, obs_data, ode_data, **params):
        r"""
        The loglikelihood of ``p(obs_data | ode_data, **params)``. Dependent on example. Please implement this.

        """
        pass

    def _logpost(self, ode_data, upars, **params):
        "Calculate logposterior."
        ll = self.obs_loglik(self._obs_data, ode_data[self._obs_ind], **params)
        lpi = self.logprior(upars)
        return ll + lpi

    def _solve(self, key, ode_weight, ode_init, prior_weight, prior_var, **params):
        r"""
        Solve the ODE using the Chkrebtii solver given the input arguments.

        Args:
            key (PRNGKey): PRNG key.
            ode_weight (ndarray(n_block, n_bmeas, n_bstate)): Weight matrix defining the measure prior; :math:`W`.
            ode_init (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
            prior_weight (ndarray(n_block, n_bstate, n_bstate)): Weight matrix defining the solution prior; :math:`Q`.
            prior_var (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
            params (optional): Parameters for the inference problem using keyword arguments.

        Return:
            (ndarray(n_block, n_bstate, n_bmeas)): Solution for the ODE.
        """

        Xt = rodeo.solve_sim(
            key=key,
            # define ode
            ode_fun=self._ode_fun,
            ode_weight=ode_weight,
            ode_init=ode_init,
            t_min=self._t_min,
            t_max=self._t_max,
            # solver parameters
            n_steps=self._n_steps,
            interrogate=rodeo.interrogate.interrogate_chkrebtii,
            prior_weight=prior_weight,
            prior_var=prior_var,
            **params
        )
        return Xt

    def initialize(self, key, upars_init):
        r"""
        Compute the initial loglikelihood of ``upars``.

        Args:
            key (PRNGKey): PRNG key.
            upars_init (JAX Pytree): Initial parameters for the inference problem.

        Return:
            (tuple):
            - **upars_init** (n_theta): Initial parameters for the inference problem.
            - **Xt_init** (ndarray(n_block, n_bstate, n_bmeas)): Solution for given ``upars_init`` using the Chkrebtii solver.
            - **lp_init** (float): Loglikelihood of the ``upars_init``.

        """
        dt = (self._t_max - self._t_min)/self._n_steps
        ode_weight, ode_init, prior_weight, prior_var, params = self.parse_pars(upars_init, dt)
        Xt_init = self._solve(key=key,
                              ode_weight=ode_weight, 
                              ode_init=ode_init, 
                              prior_weight=prior_weight,
                              prior_var=prior_var,
                              **params)
        lp_init = self._logpost(Xt_init, upars_init, **params)
        return (upars_init, Xt_init, lp_init)

    def step(self, key, state, **prop_params):
        r"""
        Compute one step of the MCMC algorithm given the current state.

        Args:
            key (PRNGKey): PRNG key.
            state (tuple): Current state which constains the current parameter ``upars``, ODE solution and loglikelihood.
            prop_params (optioanl): Parameters used in the proposal distribution.

        Return:
            (tuple):
            - **state** (tuple): Next state.
            - **sample** (tuple): A sample of ``upars`` using the MCMC algorithm.
        """
        keys = jax.random.split(key, num=3)
        upars_curr, Xt_curr, lp_curr = state
        upars_prop = self.prop_sim(keys[0], upars_curr, **prop_params)
        _, Xt_prop, lp_prop = self.initialize(keys[1], upars_prop)
        lacc_prop = lp_prop - self.prop_lpdf(upars_prop, upars_curr, **prop_params)
        lacc_curr = lp_curr - self.prop_lpdf(upars_curr, upars_prop, **prop_params)
        mh_acc = jnp.exp(lacc_prop - lacc_curr)
        U = jax.random.uniform(keys[2])

        def _true_fun():
            return upars_prop, Xt_prop, lp_prop

        def _false_fun():
            return state

        accept = U <= mh_acc
        state = jax.lax.cond(accept, _true_fun, _false_fun)

        sample = {
            "Theta": state[0],
            "accept": accept
        }
        return state, sample
