r"""
Ad-hoc experiment for moment-matching for the model

.. math::

    Y_i \sim pois(\exp(b0+b1x))
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.kalmantv import *
from rodeo.dalton import _forecast_update

def loglikehood_mm(key, fun, W, x0, theta,
                   tmin, tmax, n_res,
                   trans_state, mean_state, var_state,
                   fun_obs, y_obs, 
                   interrogate):
    r"""
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
        (float): Loglikelihood of :math:`p(y_{0:M} \mid p(z_{0:N}))`.

    """
    # Reshaping y_obs to be in blocks 
    n_obs, n_block, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_block, n_bobs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)

    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    trans_obs = jnp.zeros((n_block, n_bobs, n_bstate))
    mean_state_init = x0
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # split keys
    key1, key2 = jax.random.split(key)

    # lax.scan setup
    # scan function
    def scan_zy(carry, args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        logdens = carry["logdens"]
        key, subkey = jax.random.split(carry["key"])
        t = args['t']
        y_curr = args["y_obs"]
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
        trans_meas, mean_meas, var_meas = interrogate(
            key=subkey,
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
            mean_obs = fun_obs(mean_state_pred)
            var_obs = jnp.expand_dims(mean_obs, -1)
            mean_meas_obs = jnp.concatenate([mean_meas, mean_obs], axis=1)
            var_meas_obs = jax.vmap(lambda b: jsp.linalg.block_diag(var_meas[b], var_obs[b]))(jnp.arange(n_block))
            x_meas_obs = jnp.concatenate([x_meas, y_curr], axis=1)
            W_obs = jnp.concatenate([W, trans_obs], axis=1)
            logp, mean_state_next, var_state_next = jax.vmap(lambda b:
                _forecast_update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    W=W_obs[b],
                    x_meas=x_meas_obs[b],
                    mean_meas=mean_meas_obs[b],
                    trans_meas=trans_meas_obs[b],
                    var_meas=var_meas_obs[b]
                )                                                
            )(jnp.arange(n_block))
            return jnp.sum(logp), mean_state_next, var_state_next

        # only z is observed
        def z_update():
            logp, mean_state_next, var_state_next = jax.vmap(lambda b:
                _forecast_update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    W=W[b],
                    x_meas=x_meas[b],
                    mean_meas=mean_meas[b],
                    trans_meas=trans_meas[b],
                    var_meas=var_meas[b]
                )                                                
            )(jnp.arange(n_block))
            # return jnp.sum(logp), mean_state_next, var_state_next
            return jnp.sum(logp), mean_state_next, var_state_next

        logp, mean_state_next, var_state_next = jax.lax.cond(jnp.isnan(y_curr).any(), z_update, zy_update)
        logdens += logp
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next),
            "logdens": logdens,
            "key": key
        }
        return carry, carry
    
    # lax.scan setup
    # scan function
    def scan_z(carry, t):
        mean_state_filt, var_state_filt = carry["state_filt"]
        logdens = carry["logdens"]
        key, subkey = jax.random.split(carry["key"])
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
        # model interrogation
        trans_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        # kalman forecast and update
        logp, mean_state_next, var_state_next = jax.vmap(lambda b:
            _forecast_update(
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                W=W[b],
                x_meas=x_meas[b],
                mean_meas=mean_meas[b],
                trans_meas=trans_meas[b],
                var_meas=var_meas[b]
            )                                                
        )(jnp.arange(n_block))
        logdens += jnp.sum(logp)
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next),
            "logdens": logdens,
            "key": key
        }
        return carry, carry
    
    # scan p(y, z|x)
    # compute log-density of p(y_0 |x_0)
    mean_obs = fun_obs(x0)
    var_obs = jnp.expand_dims(mean_obs, -1)
    logdens_zy = jnp.sum(
        jax.vmap(lambda b:
                 jsp.stats.multivariate_normal.logpdf(y_obs[0][b], mean=trans_obs[b].dot(x0[b]) + mean_obs[b], cov=var_obs[b])
        )(jnp.arange(n_block)))
    # scan initial value
    scan_init_zy = {
        "state_filt": (mean_state_init, var_state_init),
        "logdens": logdens_zy,
        "key": key1
    }
    # args for scan
    scan_args = {
        't': jnp.arange(n_steps),
        'y_obs': y_obs[1:]
    }
    # scan itself
    zy_out, zy_out2 = jax.lax.scan(scan_zy, scan_init_zy, scan_args)
    
    # scan p(z|x)
    # scan initial value
    scan_init_z= {
        "state_filt": (mean_state_init, var_state_init),
        "logdens": 0.0,
        "key": key2
    }
    # scan itself
    z_out, z_out2 = jax.lax.scan(scan_z, scan_init_z, jnp.arange(n_steps))
    # append initial values to front
    zy_out2["state_filt"] = (
        jnp.concatenate([mean_state_init[None], zy_out2["state_filt"][0]]),
        jnp.concatenate([var_state_init[None], zy_out2["state_filt"][1]])
    )
    z_out2["state_filt"] = (
        jnp.concatenate([mean_state_init[None], z_out2["state_filt"][0]]),
        jnp.concatenate([var_state_init[None], z_out2["state_filt"][1]])
    )
    return zy_out2["logdens"][-1] - z_out2["logdens"][-1]
