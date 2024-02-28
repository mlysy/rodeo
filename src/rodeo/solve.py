r"""
Stochastic block solver for ODE initial value problems.

The ODE-IVP to be solved is defined as

.. math:: W X_t = f(X_t, t, \theta)

on the time interval :math:`t \in [a, b]` with initial condition :math:`X_a = x_0`.  In the paper, we use the notation :math:`a = t_{\mathrm{min}}` and :math:`b = t_{\mathrm{max}}`.

The stochastic solver proceeds via Kalman filtering and smoothing of "interrogations" of the ODE model as described in Chkrebtii et al 2016, Schober et al 2019.  In the context of the underlying Kalman filterer/smoother, the Gaussian state-space model is

.. math::

    X_n = c_n + Q_n x_{n-1} + R_n^{1/2} \epsilon_n

    Z_n = W_n X_n - f(X_n, t, \theta) + V_n^{1/2} \eta_n.

We assume that :math:`c_n = c, Q_n = Q, R_n = R`, and :math:`W_n = W` for all :math:`n`.

This module optimizes the calculations when :math:`Q`, :math:`R`, and :math:`W`, are block diagonal matrices of conformable and "stackable" sizes.  That is, recall that the dimension of these matrices are `n_state x n_state`, `n_state x n_state`, and `n_meas x n_state`, respectively.  Then suppose that :math:`Q` and :math:`R` consist of `n_block` blocks of size `n_bstate x n_bstate`, where `n_bstate = n_state/n_block`, and :math:`W` consists of `n_block` blocks of size `n_bmeas x n_bstate`, where `n_bmeas = n_meas/n_block`.  Then :math:`Q`, :math:`R`, :math:`W` can be stored as 3D arrays of size `n_block x n_bstate x n_bstate` and `n_block x n_bmeas x n_bstate`.  It is under this paradigm that the `ode` module operates.

"""


import jax
import jax.numpy as jnp
from rodeo.kalmantv import *


def _solve_filter(key, ode_fun, ode_weight, ode_init,
                  t_min, t_max, n_steps,
                  interrogate,
                  prior_weight, prior_var,
                  **params):
    r"""
    Forward pass of the ODE solver. Same arguments as :func:`~ode.solve_mv`.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = ode_weight.shape

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    mean_state = jnp.zeros((n_block, n_bstate))
    mean_state_init = ode_init
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # forward pass
    def scan_fun(carry, t):
        mean_state_filt, var_state_filt = carry["state_filt"]
        key, subkey = jax.random.split(carry["key"])
        # kalman predict
        mean_state_pred, var_state_pred = jax.vmap(predict)(
                mean_state_past=mean_state_filt,
                var_state_past=var_state_filt,
                mean_state=mean_state,
                wgt_state=prior_weight,
                var_state=prior_var
        )
        # model interrogation
        wgt_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=t_min + (t_max-t_min)*(t+1)/n_steps,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            **params
        )
        W_meas = ode_weight + wgt_meas
        # kalman update
        mean_state_next, var_state_next = jax.vmap(update)(
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                x_meas=x_meas,
                mean_meas=mean_meas,
                wgt_meas=W_meas,
                var_meas=var_meas
        )
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next),
            "key": key
        }
        stack = {
            "state_filt": (mean_state_next, var_state_next),
            "state_pred": (mean_state_pred, var_state_pred)
        }
        return carry, stack
    # scan initial value
    scan_init = {
        "state_filt": (mean_state_init, var_state_init),
        "key": key
    }
    # scan itself
    _, scan_out = jax.lax.scan(scan_fun, scan_init, jnp.arange(n_steps))
    # append initial values to front
    scan_out["state_filt"] = (
        jnp.concatenate([mean_state_init[None], scan_out["state_filt"][0]]),
        jnp.concatenate([var_state_init[None], scan_out["state_filt"][1]])
    )
    scan_out["state_pred"] = (
        jnp.concatenate([mean_state_init[None], scan_out["state_pred"][0]]),
        jnp.concatenate([var_state_init[None], scan_out["state_pred"][1]])
    )
    return scan_out


def solve_sim(key, ode_fun, ode_weight, ode_init,
              t_min, t_max, n_steps,
              interrogate,
              prior_weight, prior_var,
              **params):
    r"""
    Draw sample solution. Same arguments as :func:`~ode.solve_mv`.

    Returns:
        (ndarray(n_steps+1, n_blocks, n_bstate)): Sample solution for :math:`X_t` at times :math:`t \in [a, b]`.

    """
    n_block = prior_weight.shape[0]
    key, *subkeys = jax.random.split(key, num=n_steps+1)
    # subkeys = jnp.reshape(jnp.array(subkeys), newshape=(n_steps, n_block, 2))

    # forward pass
    filt_out = _solve_filter(
        key=key,
        ode_fun=ode_fun, ode_weight=ode_weight, ode_init=ode_init,
        t_min=t_min, t_max=t_max, n_steps=n_steps,
        interrogate=interrogate,
        prior_weight=prior_weight, prior_var=prior_var,
        **params
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    def scan_fun(x_state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        key = smooth_kwargs['key']

        mean_state_sim, var_state_sim = jax.vmap(smooth_sim)(
            x_state_next=x_state_next,
            wgt_state=prior_weight,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        x_state_curr = jax.random.multivariate_normal(key, mean_state_sim, var_state_sim, method='svd')
        return x_state_curr, x_state_curr
    # initialize
    scan_init = jax.random.multivariate_normal(
        subkeys[n_steps-1],
        mean_state_filt[n_steps],
        var_state_filt[n_steps],
        method='svd')
    
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1],
        'key': jnp.array(subkeys[:n_steps-1])
    }
    # Note: initial value x0 is assumed to be known, so we don't
    # sample it.  In fact, doing so would probably fail due to cholesky
    # of a zero variance matrix...
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    x_state_smooth = jnp.concatenate(
        [ode_init[None], scan_out, scan_init[None]]
    )
    return x_state_smooth


def solve_mv(key, ode_fun, ode_weight, ode_init,
             t_min, t_max, n_steps,
             interrogate,
             prior_weight, prior_var,
             **params):
    r"""
    Mean and variance of the stochastic ODE solver.

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
        params (kwargs): Optional model parameters.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate, _ = prior_weight.shape
    # forward pass
    filt_out = _solve_filter(
        key=key,
        ode_fun=ode_fun, ode_weight=ode_weight, ode_init=ode_init,
        t_min=t_min, t_max=t_max, n_steps=n_steps,
        interrogate=interrogate,
        prior_weight=prior_weight, prior_var=prior_var,
        **params
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        mean_state_curr, var_state_curr = jax.vmap(smooth_mv)(
                mean_state_next=state_next["mean"],
                var_state_next=state_next["var"],
                wgt_state=prior_weight,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
        )
        state_curr = {
            "mean": mean_state_curr,
            "var": var_state_curr
        }
        return state_curr, state_curr
    # initialize
    scan_init = {
        "mean": mean_state_filt[n_steps],
        "var": var_state_filt[n_steps]
    }
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    mean_state_smooth = jnp.concatenate(
        [ode_init[None], scan_out["mean"], scan_init["mean"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_block, n_bstate, n_bstate))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return mean_state_smooth, var_state_smooth
