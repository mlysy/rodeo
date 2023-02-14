r"""
This module implements the double filter solver.

The model is

.. math::

    x_0 = v

    X_n = Q X_{n-1} + R^{1/2} \epsilon_n

    z_n = W X_n - f(X_n, t_n) + V_n^{1/2} \eta_n
    
    y_n = D X_n + \Omega \zeta_n.

"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.kalmantv import *
from rodeo.ode import interrogate_rodeo

# use interrogations and observations
def forward(key, fun, W, x0, theta,
            tmin, tmax, n_res,
            trans_state, mean_state, var_state,
            trans_obs, mean_obs, var_obs, y_obs):
    r"""
    Forward pass of the double filter algorithm.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Reshaping y_obs to be in blocks 
    n_obs, n_block, n_bmeas = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_block, n_bmeas))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)

    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    mean_state_init = x0
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # lax.scan setup
    # scan function
    def scan_fun(carry, args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        t = args['t']
        y_curr = args['y_obs']
        # kalman predict
        mean_state_pred, var_state_pred = jax.vmap(lambda b:
            predict(
                mean_state_past=mean_state_filt[b],
                var_state_past=var_state_filt[b],
                mean_state=mean_state[b],
                trans_state=trans_state[b],
                var_state=var_state[b]
            )
        )(jnp.arange(n_block))
        # compute meas parameters
        trans_meas, mean_meas, var_meas = interrogate_rodeo(
            key=key,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        # both z and y are observed
        def zy_update():
            trans_meas_obs = jnp.concatenate([trans_meas, trans_obs], axis=1)
            mean_meas_obs = jnp.concatenate([mean_meas, mean_obs], axis=1)
            # var_meas_obs = jnp.concatenate([var_meas, var_obs], axis=1)
            var_meas_obs = jax.vmap(lambda b: jsp.linalg.block_diag(var_meas[b], var_obs[b]))(jnp.arange(n_block))
            x_meas_obs = jnp.concatenate([x_meas, y_curr], axis=1)
            W_obs = jnp.concatenate([W, trans_obs], axis=1)
            # kalman update
            mean_state_next, var_state_next = jax.vmap(lambda b:
                update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    W=W_obs[b],
                    x_meas=x_meas_obs[b],
                    mean_meas=mean_meas_obs[b],
                    trans_meas=trans_meas_obs[b],
                    var_meas=var_meas_obs[b]
                )
            )(jnp.arange(n_block))
            return mean_state_next, var_state_next
        # only z is observed
        def z_update():
            # kalman update
            mean_state_next, var_state_next = jax.vmap(lambda b:
                update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    W=W[b],
                    x_meas=x_meas[b],
                    mean_meas=mean_meas[b],
                    trans_meas=trans_meas[b],
                    var_meas=var_meas[b]
                )
            )(jnp.arange(n_block))
            return mean_state_next, var_state_next

        mean_state_next, var_state_next = jax.lax.cond(jnp.isnan(y_curr).any(), z_update, zy_update)
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next)
        }
        stack = {
            "state_filt": (mean_state_next, var_state_next),
            "state_pred": (mean_state_pred, var_state_pred)
        }
        return carry, stack

    # scan initial value
    scan_init = {
        "state_filt": (mean_state_init, var_state_init),
    }
    # args for scan
    scan_args = {
        't': jnp.arange(n_steps),
        'y_obs': y_obs[1:]
    }

    # scan itself
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_args)
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
