r"""
This module implements the DALTON solver which gives an approximate likelihood of :math:`p(y_{0:M} \mid z_{0:N})`.

The model is

.. math::

    x_0 = v

    X_n = c_n + Q_n X_{n-1} + R_n^{1/2} \epsilon_n

    z_n = W_n X_n - f(X_n, t_n) + V_n^{1/2} \eta_n
    
    y_m = g(X_m, \phi_m)

where :math:`g()` is a general distribution function. In the case that :math:`g()` is Gaussian, use :func:`~dalton.loglikehood` for a better approximation. In other cases, use :func:`~dalton.loglikehood_nn`. We assume that :math:`c_n = c, Q_n = Q, R_n = R`, and :math:`W_n = W` for all :math:`n`.

In the Gaussian case, we assume the observation model is

.. math::

    y_m = D_m X_m + \Omega^{1/2} \epsilon_m.

We assume that the :math:`M \leq N`, so that the observation step size is larger than that of the evaluation step size.

"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.kalmantv import *
from rodeo.utils import multivariate_normal_logpdf
import rodeo.ode as rode

# use linearizations and observations
def _solve_filter(key, fun, W, x0, theta,
                  tmin, tmax, n_res,
                  wgt_state, var_state,
                  wgt_obs, var_obs, y_obs, 
                  interrogate):
    r"""
    Forward pass of the DALTON algorithm with Gaussian observations.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Resolution number determining how to thin solution process to match observations.
        wgt_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        wgt_obs (ndarray(n_obs, n_block, n_bobs, n_state)): Transition matrix defining the noisy observations; :math:`D`.
        var_obs (ndarray(n_block, n_bobs, n_bobs)): Variance matrix defining the noisy observations; :math:`\Omega`.
        y_obs (ndarray(n_obs, n_block, n_bobs)): Observed data; :math:`y_{0:M}`.
        interrogate (function): Function defining the linearization method.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape
    n_obs, _, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    mean_obs = jnp.zeros((n_block, n_bobs))
    mean_state = jnp.zeros((n_block, n_bstate))
    mean_state_init = x0
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # lax.scan setup
    # scan function
    def scan_fun(carry, args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        key, subkey = jax.random.split(carry["key"])
        t = args['t']
        i = (t+1)//n_res
        # kalman predict
        mean_state_pred, var_state_pred = jax.vmap(lambda b:
            predict(
                mean_state_past=mean_state_filt[b],
                var_state_past=var_state_filt[b],
                mean_state=mean_state[b],
                wgt_state=wgt_state[b],
                var_state=var_state[b]
            )
        )(jnp.arange(n_block))
        # compute meas parameters
        wgt_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        W_meas = W + wgt_meas
        # both z and y are observed
        def zy_update():
            y_curr = y_obs[i]
            wgt_curr = wgt_obs[i]
            wgt_meas_obs = jnp.concatenate([W_meas, wgt_curr], axis=1)
            mean_meas_obs = jnp.concatenate([mean_meas, mean_obs], axis=1)
            var_meas_obs = jax.vmap(lambda b: jsp.linalg.block_diag(var_meas[b], var_obs[b]))(jnp.arange(n_block))
            x_meas_obs = jnp.concatenate([x_meas, y_curr], axis=1)
            # kalman update
            mean_state_next, var_state_next = jax.vmap(lambda b:
                update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    x_meas=x_meas_obs[b],
                    mean_meas=mean_meas_obs[b],
                    wgt_meas=wgt_meas_obs[b],
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
                    x_meas=x_meas[b],
                    mean_meas=mean_meas[b],
                    wgt_meas=W_meas[b],
                    var_meas=var_meas[b]
                )
            )(jnp.arange(n_block))
            return mean_state_next, var_state_next

        mean_state_next, var_state_next = jax.lax.cond(i*n_res == (t+1), zy_update, z_update)
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next),
            "key" : key
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
    # args for scan
    scan_args = {
        't': jnp.arange(n_steps)
        # 'y_obs': y_obs[1:]
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

def solve_mv(key, fun, W, x0, theta,
             tmin, tmax, n_res,
             wgt_state, var_state,
             wgt_obs, var_obs, y_obs, 
             interrogate):
    r"""
    DALTON algorithm to compute the mean and variance of :math:`X_{0:N}` assuming Gaussian observations.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Determines number of evaluations between observations; resolution number.
        wgt_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        wgt_obs (ndarray(n_obs, n_block, n_bobs, n_state)): Transition matrix defining the noisy observations; :math:`D`.
        var_obs (ndarray(n_block, n_bobs, n_bobs)): Variance matrix defining the noisy observations; :math:`\Omega`.
        y_obs (ndarray(n_obs, n_block, n_bobs)): Observed data; :math:`y_{0:M}`.
        interrogate (function): Function defining the linearization method.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    # Reshaping y_obs to be in blocks 
    n_block = wgt_state.shape[0]

    # get dimensions
    n_block, n_bstate, _ = wgt_state.shape
    # forward pass
    filt_out = _solve_filter(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_res=n_res, wgt_state=wgt_state,
        var_state=var_state,
        wgt_obs=wgt_obs,
        var_obs=var_obs, y_obs=y_obs,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]
    n_steps = len(mean_state_pred) - 1
    # backward pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        mean_state_curr, var_state_curr = jax.vmap(lambda b:
            smooth_mv(
                mean_state_next=state_next["mean"][b],
                var_state_next=state_next["var"][b],
                wgt_state=wgt_state[b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
            )
        )(jnp.arange(n_block))
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
        [x0[None], scan_out["mean"], scan_init["mean"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_block, n_bstate, n_bstate))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return mean_state_smooth, var_state_smooth

def _forecast_update(mean_state_pred, var_state_pred,
                     x_meas, mean_meas,
                     wgt_meas, var_meas):
    r"""
    Perform one update step of the Kalman filter and forecast.

    Args:
        mean_state_pred (ndarray(n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_block, n_bstate, n_sbtate)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Matrix for getting the derivative; denoted by :math:`W`.
        x_meas (ndarray(n_block, n_bmeas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mean_meas (ndarray(n_block, n_bmeas)): Transition offsets defining the measure prior; denoted by :math:`c`.
        wgt_meas (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; denoted by :math:`W+B`.
        var_meas (ndarray(n_block, n_bmeas, n_bmeas)): Variance matrix defining the measure prior; denoted by :math:`\Sigma_n`.

    Returns:
        (tuple):
        - **logdens** (float): The log-likelihood for the observations.
        - **mean_state_filt** (ndarray(n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n}`.
        - **var_state_filt** (ndarray(n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
    """
    # kalman forecast
    mean_state_fore, var_state_fore = forecast(
        mean_state_pred = mean_state_pred,
        var_state_pred = var_state_pred,
        mean_meas = mean_meas,
        wgt_meas = wgt_meas,
        var_meas = var_meas
    )
    logdens = multivariate_normal_logpdf(x_meas, mean=mean_state_fore, cov=var_state_fore)
    # kalman update
    mean_state_filt, var_state_filt = update(
        mean_state_pred = mean_state_pred,
        var_state_pred = var_state_pred,
        x_meas = x_meas,
        mean_meas = mean_meas,
        wgt_meas = wgt_meas,
        var_meas = var_meas
    )
    return logdens, mean_state_filt, var_state_filt

def dalton(key, fun, W, x0, theta,
           tmin, tmax, n_res,
           wgt_state, var_state,
           wgt_obs, var_obs, y_obs, 
           interrogate):
    r"""
    Compute marginal loglikelihood of DALTON algorithm for Gaussian observations; :math:`p(y_{0:M} \mid z_{0:N})`.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Resolution number determining how to thin solution process to match observations.
        wgt_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        wgt_obs (ndarray(n_obs, n_block, n_bobs, n_state)): Transition matrix defining the noisy observations; :math:`D`.
        var_obs (ndarray(n_block, n_bobs, n_bobs)): Variance matrix defining the noisy observations; :math:`\Omega`.
        y_obs (ndarray(n_obs, n_block, n_bobs)): Observed data; :math:`y_{0:M}`.
        interrogate (function): Function defining the linearization method.

    Returns:
        (float): Loglikelihood of :math:`p(y_{0:M} \mid z_{0:N})`.

    """
    # Reshaping y_obs to be in blocks 
    n_obs, _, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res

    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    mean_obs = jnp.zeros((n_block, n_bobs))
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
                wgt_state=wgt_state[b],
                var_state=var_state[b]
            )
        )(jnp.arange(n_block))
        # compute meas parameters
        wgt_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        W_meas = wgt_meas + W
        # both z and y are observed
        def zy_update():
            y_curr = y_obs[i]
            wgt_curr = wgt_obs[i]
            wgt_meas_obs = jnp.concatenate([W_meas, wgt_curr], axis=1)
            mean_meas_obs = jnp.concatenate([mean_meas, mean_obs], axis=1)
            var_meas_obs = jax.vmap(lambda b: jsp.linalg.block_diag(var_meas[b], var_obs[b]))(jnp.arange(n_block))
            x_meas_obs = jnp.concatenate([x_meas, y_curr], axis=1)
            logp, mean_state_next, var_state_next = jax.vmap(lambda b:
                _forecast_update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    x_meas=x_meas_obs[b],
                    mean_meas=mean_meas_obs[b],
                    wgt_meas=wgt_meas_obs[b],
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
                    wgt_meas=W_meas[b],
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
                wgt_state=wgt_state[b],
                var_state=var_state[b]
            )
        )(jnp.arange(n_block))
        # model linearization
        wgt_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        W_meas = W + wgt_meas
        # kalman forecast and update
        logp, mean_state_next, var_state_next = jax.vmap(lambda b:
            _forecast_update(
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                x_meas=x_meas[b],
                mean_meas=mean_meas[b],
                wgt_meas=W_meas[b],
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
    logdens_zy = jnp.sum(
        jax.vmap(lambda b:
                 multivariate_normal_logpdf(y_obs[0][b], mean=wgt_obs[0][b].dot(x0[b]) + mean_obs[b], cov=var_obs[b])
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

# --------------------------------------------- Non Gaussian Observations ------------------------------------------------------------------

def _solve_filter_nn(key, fun, W, x0, theta,
                     tmin, tmax, n_res,
                     wgt_state, var_state,
                     fun_obs, wgt_obs, y_obs, 
                     interrogate):
    r"""
    Forward pass of the DALTON algorithm using non-Gaussian observations.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Resolution number determining how to thin solution process to match observations.
        wgt_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        fun_obs (function): Observation likelihood function.
        wgt_obs (ndarray(n_obs, n_block, n_bobs, n_bstate)): Transition matrix defining the noisy observations; :math:`D`.
        y_obs (ndarray(n_obs, n_yblock, n_bobs)): Observed data; :math:`y_{0:M}`.
        interrogate (function): Function defining the linearization method.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state, measure and observation variables
    n_block, n_bmeas, n_bstate = W.shape
    n_obs, _, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    mean_obs = jnp.zeros((n_block, n_bobs))
    mean_state = jnp.zeros((n_block, n_bstate))
    mean_state_init = x0
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # lax.scan setup
    # scan function
    def scan_fun(carry, args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        key, subkey = jax.random.split(carry["key"])
        t = args['t']
        i = (t+1)//n_res
        # y_curr = args['y_obs']
        # kalman predict
        mean_state_pred, var_state_pred = jax.vmap(lambda b:
            predict(
                mean_state_past=mean_state_filt[b],
                var_state_past=var_state_filt[b],
                mean_state=mean_state[b],
                wgt_state=wgt_state[b],
                var_state=var_state[b]
            )
        )(jnp.arange(n_block))
        # compute meas parameters
        wgt_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        W_meas = W + wgt_meas
        # both z and y are observed
        def zy_update():
            y_curr = y_obs[i]
            wgt_curr = wgt_obs[i]
            # transform to yhat
            Cmu = jax.vmap(lambda b: wgt_curr[b].dot(mean_state_pred[b]))(jnp.arange(n_block))
            gpmu = jax.jacfwd(fun_obs)(Cmu, y_curr, theta, i)
            gppmu = jax.jacfwd(jax.jacrev(fun_obs))(Cmu, y_curr, theta, i)
            var_obs = jax.vmap(lambda b: -jnp.linalg.pinv(gppmu[b, :, b]))(jnp.arange(n_block))
            y_new = jax.vmap(lambda b: Cmu[b] + var_obs[b].dot(gpmu[b]))(jnp.arange(n_block))
            # stack measure and observation variables
            wgt_meas_obs = jnp.concatenate([W_meas, wgt_curr], axis=1)
            mean_meas_obs = jnp.concatenate([mean_meas, mean_obs], axis=1)
            var_meas_obs = jax.vmap(lambda b: jsp.linalg.block_diag(var_meas[b], var_obs[b]))(jnp.arange(n_block))
            x_meas_obs = jnp.concatenate([x_meas, y_new], axis=1)
            # kalman update
            mean_state_next, var_state_next = jax.vmap(lambda b:
                update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    x_meas=x_meas_obs[b],
                    mean_meas=mean_meas_obs[b],
                    wgt_meas=wgt_meas_obs[b],
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
                    x_meas=x_meas[b],
                    mean_meas=mean_meas[b],
                    wgt_meas=W_meas[b],
                    var_meas=var_meas[b]
                )
            )(jnp.arange(n_block))
            return mean_state_next, var_state_next

        mean_state_next, var_state_next = jax.lax.cond(i*n_res == (t+1), zy_update, z_update)
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next),
            "key" : key
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
    # args for scan
    scan_args = {
        't': jnp.arange(n_steps)
        # 'y_obs': y_obs[1:]
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

def solve_mv_nn(key, fun, W, x0, theta,
                tmin, tmax, n_res,
                wgt_state, var_state,
                fun_obs, wgt_obs, y_obs, 
                interrogate):
    r"""
    DALTON algorithm to compute the mean and variance of :math:`X_{0:N}` assuming non-Gaussian observations.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Determines number of evaluations between observations; resolution number.
        wgt_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        fun_obs (function): Observation likelihood function.
        wgt_obs (ndarray(n_obs, n_block, n_bobs, n_bstate)): Transition matrix defining the noisy observations; :math:`D`.
        y_obs (ndarray(n_obs, n_yblock, n_bobs)): Observed data; :math:`y_{0:M}`.
        interrogate (function): Function defining the linearization method.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    # get dimensions
    n_block, n_bstate, _ = wgt_state.shape
    # Reshaping y_obs to be in blocks 
    n_obs = y_obs.shape[0]
    n_steps = (n_obs-1)*n_res

    # forward pass
    filt_out = _solve_filter_nn(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_res=n_res, wgt_state=wgt_state,
        var_state=var_state,
        fun_obs=fun_obs, wgt_obs=wgt_obs, y_obs=y_obs,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]
    # backward pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        mean_state_curr, var_state_curr = jax.vmap(lambda b:
            smooth_mv(
                mean_state_next=state_next["mean"][b],
                var_state_next=state_next["var"][b],
                wgt_state=wgt_state[b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
            )
        )(jnp.arange(n_block))
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
        [x0[None], scan_out["mean"], scan_init["mean"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_block, n_bstate, n_bstate))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return mean_state_smooth, var_state_smooth

def _logx_yhat(mean_state_filt, var_state_filt,
               mean_state_pred, var_state_pred,
               wgt_state):
    r"""
    Compute the loglikelihood of :math:`p(X_{0:N} \mid \hat y_{0:M}, z_{0:N}=0)`.
    
    Args:
        mean_state_pred (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        var_state_pred (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        mean_state_filt (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        var_state_filt (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        wgt_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **logx_yhat** (float): Loglikelihood of :math:`p(X_{0:N} \mid \hat y_{0:M}, z_{0:N}=0)`.
    """
    # dimensions
    n_tot, n_block, n_bstate = mean_state_filt.shape
    n_steps = n_tot - 1

    # backward pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        logx_yhat = state_next["logx_yhat"]
        mean_state_curr, var_state_curr = jax.vmap(lambda b:
            smooth_mv(
                mean_state_next=state_next["mean"][b],
                var_state_next=state_next["var"][b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                wgt_state=wgt_state[b]
            )
        )(jnp.arange(n_block))
        mean_state_sim, var_state_sim = jax.vmap(lambda b:
            smooth_sim(
                x_state_next=state_next["mean"][b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                wgt_state=wgt_state[b]
            )
        )(jnp.arange(n_block))
        logx_yhat += jnp.sum(
            jax.vmap(lambda b:
                     multivariate_normal_logpdf(mean_state_curr[b], mean=mean_state_sim[b], cov=var_state_sim[b])
                    )(jnp.arange(n_block))
        )
        state_curr = {
            "mean": mean_state_curr,
            "var": var_state_curr,
            "logx_yhat":  logx_yhat
        }
        return state_curr, state_curr
    # compute log(mu_{N|N}) at the last filtering step
    logx_yhat0 = jnp.sum(
        jax.vmap(lambda b:
                 multivariate_normal_logpdf(mean_state_filt[n_steps][b], mean=mean_state_filt[n_steps][b], cov=var_state_filt[n_steps][b])
                )(jnp.arange(n_block)))
    # initialize
    scan_init = {
        "mean": mean_state_filt[n_steps],
        "var": var_state_filt[n_steps],
        "logx_yhat": logx_yhat0
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
    scan_out["mean"] = jnp.concatenate(
        [mean_state_filt[0][None], scan_out["mean"], scan_init["mean"][None]]
    )
    return scan_out["mean"], scan_out["logx_yhat"]

def _logx_z(uncond_mean, 
            mean_state_filt, var_state_filt,
            mean_state_pred, var_state_pred,
            wgt_state):
    r"""
    Compute the loglikelihood of :math:`p(X_{0:N} \mid z_{0:N}=0)`.
    
    Args:
        uncond_mean (ndarray(n_steps+1, n_block, n_bstate)): Unconditional mean computed from :math:`p(X_{0:N} \mid \hat y_{0:M}, z_{0:N}=0)`.
        mean_state_pred (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        var_state_pred (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        mean_state_filt (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        var_state_filt (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        wgt_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
    
    Return:
        (float): Loglikelihood of :math:`p(X_{0:N} \mid z_{0:N}=0)`.

    """
    # dimensions
    n_tot, n_block, n_bstate = mean_state_filt.shape
    n_steps = n_tot - 1

    # backward pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        uncond_next = smooth_kwargs['uncond_next']
        uncond_curr = smooth_kwargs['uncond_curr']
        logx_z = state_next["logx_z"]
        mean_state_sim, var_state_sim = jax.vmap(lambda b:
            smooth_sim(
                x_state_next=uncond_next[b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                wgt_state=wgt_state[b]
            )
        )(jnp.arange(n_block))
        logx_z += jnp.sum(
            jax.vmap(lambda b:
                     multivariate_normal_logpdf(uncond_curr[b], mean=mean_state_sim[b], cov=var_state_sim[b])
                    )(jnp.arange(n_block))
        )
        state_curr = {
            "logx_z":  logx_z
        }
        return state_curr, state_curr
    # compute log(mu_{N|N}) at the last filtering step
    logx_z0 = jnp.sum(
        jax.vmap(lambda b:
                 multivariate_normal_logpdf(uncond_mean[n_steps][b], mean=mean_state_filt[n_steps][b], cov=var_state_filt[n_steps][b])
                )(jnp.arange(n_block)))
    # initialize
    scan_init = {
        "logx_z": logx_z0,
    }
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1],
        'uncond_next': uncond_mean[2:n_steps+1],
        'uncond_curr': uncond_mean[1:n_steps]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)
    
    return scan_out["logx_z"]

def daltonng(key, fun, W, x0, theta,
             tmin, tmax, n_res,
             wgt_state, var_state,
             fun_obs, wgt_obs, y_obs, 
             interrogate):
    r"""
    Compute marginal loglikelihood of DALTON algorithm for non-Gaussian observations; :math:`p(y_{0:M} \mid z_{0:N})`.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Resolution number determining how to thin solution process to match observations.
        wgt_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        fun_obs (function): Observation likelihood function.
        wgt_obs (ndarray(n_obs, n_block, n_bobs, n_state)): Transition matrix defining the noisy observations; :math:`D`.
        y_obs (ndarray(n_obs, n_yblock, n_bobs)): Observed data; :math:`y_{0:M}`.
        interrogate (function): Function defining the linearization method.
    
    Returns:
        (float): Loglikelihood of :math:`p(y_{0:M} \mid z_{0:N})`.

    """
    # Reshaping y_obs to be in blocks 
    n_block = W.shape[0]
    n_obs = y_obs.shape[0]
    n_steps = (n_obs-1)*n_res

    # forward pass
    filt_out = _solve_filter_nn(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_res=n_res, wgt_state=wgt_state,
        var_state=var_state,
        fun_obs=fun_obs, wgt_obs=wgt_obs, y_obs=y_obs,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # logp(x | hat y, z)
    mean_state_smooth, logx_yhat = _logx_yhat(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        wgt_state=wgt_state
    )

    # logp(y | x)
    def vmap_fun(i):
        t = i*n_res
        Cx = jax.vmap(lambda b: wgt_obs[i][b].dot(mean_state_smooth[t][b]))(jnp.arange(n_block))
        return fun_obs(Cx, y_obs[i], theta, i)
    logy_x = jnp.sum(jax.vmap(vmap_fun)(jnp.arange(0, n_obs+1)))

    # logp(x | z)
    # first do forward pass without obs
    filt_out = rode._solve_filter(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_steps=n_steps, wgt_state=wgt_state,
        var_state=var_state,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    logx_z = _logx_z(
        uncond_mean=mean_state_smooth,
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        wgt_state=wgt_state
    )

    return logy_x + logx_z[0] - logx_yhat[0]
