r"""
Non-block version of rodeo.
"""

import jax
import jax.numpy as jnp
from rodeo.kalmantv.standard import *


def interrogate_rodeo(key, ode_fun, ode_weight, t,
                      mean_state_pred, var_state_pred,
                      **params):
    var_meas = jnp.atleast_2d(
        jnp.linalg.multi_dot([ode_weight, var_state_pred, ode_weight.T])
    )
    mean_meas = -ode_fun(mean_state_pred, t, **params)
    return jnp.zeros(ode_weight.shape), mean_meas, var_meas


def interrogate_chkrebtii(key, ode_fun, ode_weight, t,
                          mean_state_pred, var_state_pred,
                          **params):

    var_meas = jnp.atleast_2d(
        jnp.linalg.multi_dot([ode_weight, var_state_pred, ode_weight.T])
    )
    x_state = jax.random.multivariate_normal(key, mean_state_pred, var_state_pred)
    mean_meas = -ode_fun(x_state, t, **params)
    return jnp.zeros(ode_weight.shape), mean_meas, var_meas


def interrogate_schober(key, ode_fun, ode_weight, t,
                        mean_state_pred, var_state_pred,
                        **params):

    n_meas = ode_weight.shape[0]
    var_meas = jnp.zeros((n_meas, n_meas))
    mean_meas = -ode_fun(mean_state_pred, t, **params)
    return jnp.zeros(ode_weight.shape), mean_meas, var_meas


def interrogate_kramer(key, ode_fun, ode_weight, t,
                       mean_state_pred, var_state_pred,
                       **params):

    n_meas, n_state = ode_weight.shape
    fun_meas = -ode_fun(mean_state_pred, t, **params)
    jac = jax.jacfwd(ode_fun)(mean_state_pred, t, **params)
    wgt_meas = -jac
    mean_meas = fun_meas + jac.dot(mean_state_pred)
    var_meas = jnp.zeros((n_meas, n_meas))
    return wgt_meas, mean_meas, var_meas


def _solve_filter(key, ode_fun, ode_weight, ode_init,
                  t_min, t_max, n_steps,
                  interrogate,
                  prior_weight, prior_var,
                  **params):
    
    # Dimensions of state and measure variables
    n_meas, n_state = ode_weight.shape

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_meas))
    mean_state = jnp.zeros((n_state))
    mean_state_init = ode_init
    var_state_init = jnp.zeros((n_state, n_state))
    
    # forward pass
    def scan_fun(carry, t):
        mean_state_filt, var_state_filt = carry["state_filt"]
        key, subkey = jax.random.split(carry["key"])
        # kalman predict
        mean_state_pred, var_state_pred = predict(
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
        mean_state_next, var_state_next = update(
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
  
    key, *subkeys = jax.random.split(key, num=n_steps+1)
    subkeys = jnp.array(subkeys)

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
    # lax.scan setup
    def scan_fun(x_state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        key = smooth_kwargs['key']

        mean_state_sim, var_state_sim = smooth_sim(
            x_state_next=x_state_next,
            wgt_state=prior_weight,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        x_state_curr = jax.random.multivariate_normal(key, mean_state_sim, var_state_sim, method="svd")
        return x_state_curr, x_state_curr
    # initialize
    scan_init = jax.random.multivariate_normal(subkeys[n_steps-1], 
                                               mean_state_filt[n_steps], 
                                               var_state_filt[n_steps],
                                               method='svd')
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1],
        'key': subkeys[:n_steps-1]
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
   
    n_state = prior_weight.shape[0]
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
        mean_state_curr, var_state_curr = smooth_mv(
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
        [jnp.zeros((n_state, n_state))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return mean_state_smooth, var_state_smooth
