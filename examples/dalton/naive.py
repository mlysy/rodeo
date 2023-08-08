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
from rodeo.fenrir import forward, backward_param
from rodeo.utils import multivariate_normal_logpdf

def loglikehood_mm(key, fun, W, x0, theta,
                   tmin, tmax, n_res,
                   trans_state, var_state,
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
    n_obs, _, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res

    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    trans_obs = jnp.zeros((n_block, n_bobs, n_bstate))
    mean_state = jnp.zeros((n_block, n_bstate))
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
        i = (t+1)//n_res
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
        W_meas = trans_meas + W
        # both z and y are observed
        def zy_update():
            y_curr = y_obs[i]
            trans_meas_obs = jnp.concatenate([W_meas, trans_obs], axis=1)
            mean_obs = fun_obs(mean_state_pred, theta)
            var_obs = jnp.expand_dims(mean_obs, -1)
            mean_meas_obs = jnp.concatenate([mean_meas, mean_obs], axis=1)
            var_meas_obs = jax.vmap(lambda b: jsp.linalg.block_diag(var_meas[b], var_obs[b]))(jnp.arange(n_block))
            x_meas_obs = jnp.concatenate([x_meas, y_curr], axis=1)
            logp, mean_state_next, var_state_next = jax.vmap(lambda b:
                _forecast_update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
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
                    x_meas=x_meas[b],
                    mean_meas=mean_meas[b],
                    trans_meas=W_meas[b],
                    var_meas=var_meas[b]
                )                                                
            )(jnp.arange(n_block))
            return jnp.sum(logp), mean_state_next, var_state_next

        logp, mean_state_next, var_state_next = jax.lax.cond(i*n_res==(t+1), zy_update, z_update)
        logdens += logp
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next),
            "logdens": logdens,
            "key": key
        }
        return carry, None
    
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
        # model linearization
        trans_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        W_meas = W + trans_meas
        # kalman forecast and update
        logp, mean_state_next, var_state_next = jax.vmap(lambda b:
            _forecast_update(
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                x_meas=x_meas[b],
                mean_meas=mean_meas[b],
                trans_meas=W_meas[b],
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
        return carry, None
    
    # scan p(y, z|x)
    # compute log-density of p(y_0 |x_0)
    mean_obs = fun_obs(x0, theta)
    var_obs = jnp.expand_dims(mean_obs, -1)
    logdens_zy = jnp.sum(
        jax.vmap(lambda b:
                 multivariate_normal_logpdf(y_obs[0][b], mean=trans_obs[b].dot(x0[b]) + mean_obs[b], cov=var_obs[b])
        )(jnp.arange(n_block)))
    # scan initial value
    scan_init_zy = {
        "state_filt": (mean_state_init, var_state_init),
        "logdens": logdens_zy,
        "key": key1
    }
    # args for scan
    scan_args = {
        't': jnp.arange(n_steps)
        # 'y_obs': y_obs[1:]
    }
    # scan itself
    zy_out, _ = jax.lax.scan(scan_zy, scan_init_zy, scan_args)
    
    # scan p(z|x)
    # scan initial value
    scan_init_z= {
        "state_filt": (mean_state_init, var_state_init),
        "logdens": 0.0,
        "key": key2
    }
    # scan itself
    z_out, _ = jax.lax.scan(scan_z, scan_init_z, jnp.arange(n_steps))
    return zy_out["logdens"] - z_out["logdens"]


def backward(theta, n_res, trans_state, mean_state, var_state,
             fun_obs, y_obs):
    
    r"""
    Backward pass of Fenrir algorithm where observations are used.

    Args:
        n_res (int): Resolution number determining how to thin solution process to match observations.
        trans_state (ndarray(n_steps, n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`A_{0:N}`.
        mean_state (ndarray(n_steps+1, n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`b_{0:N}`.
        var_state (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`C_{0:N}`.
        trans_obs (ndarray(n_obs, n_block, n_bobs, n_state)): Transition matrix defining the noisy observations; :math:`D_{0:M}`.
        var_obs (ndarray(n_block, n_bobs, n_bobs)): Variance matrix defining the noisy observations; :math:`\Omega`.
        y_obs (ndarray(n_steps, n_block, n_bobs)): Observed data; :math:`y_{0:M}`.

    Returns:
        (float) : The logdensity of :math:`p(y_{0:M} \mid z_{0:N})`.

    """
    # Add point to beginning of state variable for simpler loop
    n_obs, n_block, n_bobs = y_obs.shape
    n_bstate = trans_state.shape[2]
    n_steps = (n_obs-1)*n_res
    trans_state_end = jnp.zeros(trans_state.shape[1:])
    trans_state = jnp.concatenate([trans_state, trans_state_end[None]])

    # offset of obs is assumed to be 0
    trans_obs = jnp.zeros((n_block, n_bobs, n_bstate))

    # reverse pass
    def scan_fun(carry, back_args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        logdens = carry["logdens"]
        trans_state = back_args['trans_state']
        mean_state = back_args['mean_state']
        var_state = back_args['var_state']
        t = back_args['t']
        i = t//n_res

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
        # y_obs is None
        def _no_obs():
            mean_state_filt = mean_state_pred
            var_state_filt = var_state_pred
            logp = 0.0
            return mean_state_filt, var_state_filt, logp

        # y_obs is not None
        def _obs():
            y_curr = y_obs[i]
            mean_obs = fun_obs(mean_state_pred, theta)
            var_obs = jnp.expand_dims(mean_obs, -1)
            # trans_curr = trans_obs[i]
            # kalman forecast
            mean_state_fore, var_state_fore = jax.vmap(lambda b:
                forecast(
                    mean_state_pred = mean_state_pred[b],
                    var_state_pred = var_state_pred[b],
                    mean_meas = mean_obs[b],
                    trans_meas = trans_obs[b],
                    var_meas = var_obs[b]
                    )
            )(jnp.arange(n_block))
            # logdensity of forecast
            logp = jnp.sum(jax.vmap(lambda b:
                # jsp.stats.multivariate_normal.logpdf(y_curr[b], mean=mean_state_fore[b], cov=var_state_fore[b])
                multivariate_normal_logpdf(y_curr[b], mean=mean_state_fore[b], cov=var_state_fore[b])
            )(jnp.arange(n_block)))
            # kalman update
            mean_state_filt, var_state_filt = jax.vmap(lambda b:
                update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    x_meas=y_curr[b],
                    mean_meas=mean_obs[b],
                    trans_meas=trans_obs[b],
                    var_meas=var_obs[b]
                )
            )(jnp.arange(n_block))
            return mean_state_filt, var_state_filt, logp

        mean_state_filt, var_state_filt, logp = jax.lax.cond(i*n_res == t, _obs, _no_obs)
        logdens += logp

        # output
        carry = {
            "state_filt": (mean_state_filt, var_state_filt),
            "logdens" : logdens
        }
        stack = {
            "state_pred": (mean_state_pred, var_state_pred),
            "state_filt": (mean_state_filt, var_state_filt)
        }
        return carry, stack

    # start at N+1 assuming 0 mean and variance
    scan_init = {
        "state_filt" : (jnp.zeros((n_block, n_bstate,)), jnp.zeros((n_block, n_bstate, n_bstate))),
        "logdens" : 0.0
    }
    back_args = {
        "trans_state" : trans_state,
        "mean_state" : mean_state,
        "var_state" : var_state,
        # "y_obs": y_obs
        "t": jnp.arange(n_steps+1)
    }

    scan_out, scan_out2 = jax.lax.scan(scan_fun, scan_init, back_args, reverse=True)
    return scan_out["logdens"], scan_out2

def fenrir_mm(key, fun, W, x0, theta, tmin, tmax, n_res,
              trans_state, var_state,
              fun_obs, y_obs,
              interrogate):
    
    r"""
    Fenrir algorithm to compute the approximate marginal likelihood of :math:`p(y_{0:M} \mid z_{0:N})`.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Resolution number determining how to thin solution process to match observations.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        trans_obs (ndarray(n_block, n_bobs, n_state)): Transition matrix defining the noisy observations; :math:`D`.
        var_obs (ndarray(n_block, n_bobs, n_bobs)): Variance matrix defining the noisy observations; :math:`\Omega`.
        y_obs (ndarray(n_steps, n_block, n_bobs)): Observed data; :math:`y_{0:M}`.
        interrogate (function): Function defining the linearization method.

    Returns:
        (float) : The logdensity of :math:`p(y_{0:M} \mid z_{0:N})`.

    """
    n_obs = y_obs.shape[0]
    n_steps = (n_obs-1)*n_res

    # forward pass
    filt_out = forward(
        key=key, fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, n_steps=n_steps,
        trans_state=trans_state,
        var_state=var_state,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    trans_state_cond, mean_state_cond, var_state_cond = backward_param(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        trans_state=trans_state
    )

    # reverse pass
    logdens, _ = backward(
        theta=theta, n_res=n_res, trans_state=trans_state_cond, 
        mean_state=mean_state_cond, var_state=var_state_cond, 
        fun_obs=fun_obs, y_obs=y_obs
    )

    return logdens
