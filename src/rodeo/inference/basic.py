r"""
This module implements the Basic method for computing the approximate loglikelihood of :math:`\log p(Y_{0:M} \mid Z_{1:N})`.

Using :math:`\mu_{0:N|N} = E(X_{0:N} \mid Z_{1:N})` from the rodeo solver, the approximate likelihood is computed as

.. math:: p(Y_{0:M} \mid Z_{1:N}) = \sum_{i=0}^M \log p(Y_i \mid X_{n(i)} = \mu_{n(i)|N}).

In the case that observations time grid is not the same as the solver time grid, then the observation uses the closest discretization time point.

"""
import jax
import jax.numpy as jnp
from rodeo.solve import solve_mv


def basic(key, ode_fun, ode_weight, ode_init, 
          t_min, t_max, n_steps,
          interrogate,
          prior_weight, prior_var,
          obs_data, obs_times, obs_loglik,
          **params):
    
    r"""
    Basic algorithm to compute the approximate loglikelihood of :math:`p(Y_{0:M} \mid Z_{1:N})`.

    Args:
        key (PRNGKey): PRNG key.
        ode_fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        ode_weight (ndarray(n_block, n_bmeas, n_bstate)): Weight matrix defining the measure prior; :math:`W`.
        ode_init (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        t_min (float): First time point of the time interval to be evaluated; :math:`a`.
        t_max (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        interrogate (function): Function defining the interrogation method.
        prior_weight (ndarray(n_block, n_bstate, n_bstate)): Weight matrix defining the solution prior; :math:`Q`.
        prior_var (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        obs_data (ndarray(n_obs, n_bobs)): Observed data; :math:`Y_{0:M}`.
        obs_times (ndarray(n_obs)): Observation time; :math:`0, \ldots, M`.
        obs_loglik (fun): Observation loglikelihood function.
        params (kwargs): Optional model parameters.

    Returns:
        (float) : The loglikelihood of :math:`p(Y_{0:M} \mid Z_{1:N})`.

    """

    Xt, _ = solve_mv(
        key=key,
        ode_fun=ode_fun,
        ode_weight=ode_weight,
        ode_init=ode_init,
        t_min=t_min,
        t_max=t_max,
        n_steps=n_steps,
        interrogate=interrogate,
        prior_weight=prior_weight,
        prior_var=prior_var,
        **params
    )
    sim_times = jnp.linspace(t_min, t_max, n_steps+1)
    ode_data = Xt[jnp.searchsorted(sim_times, obs_times)]
    return obs_loglik(obs_data, ode_data, **params)
