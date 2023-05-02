r"""
This module implements the pope solver.

The model is

.. math::

    x_0 = v

    X_n = Q X_{n-1} + R^{1/2} \epsilon_n

    z_n = W X_n - f(X_n, t_n) + V_n^{1/2} \eta_n
    
    y_n = C X_n + \Omega \zeta_n.

"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.kalmantv import *
import rodeo.ode as rode

def multivariate_normal_logpdf(x, mean, cov):
    """Using Eigendecomposition."""
    w, v = jnp.linalg.eigh(cov)
    z = jnp.dot(v.T, x - mean)
    z2 = z**2
    iw = ~jnp.isclose(w/jnp.max(w), 0, atol=1e-15)
    w = jnp.where(iw, w, 1.) # remove possibility of nan
    val = z2/w + jnp.log(w)
    val = -.5 * jnp.sum(jnp.where(iw, val, 0.)) - jnp.sum(iw)*.5*jnp.log(2*jnp.pi) 
    return val

# use interrogations and observations
def _solve_filter(key, fun, W, x0, theta,
                  tmin, tmax, n_steps,
                  trans_state, mean_state, var_state,
                  trans_obs, mean_obs, var_obs, y_obs, 
                  interrogate):
    r"""
    Forward pass of the pope algorithm.

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
        key, subkey = jax.random.split(carry["key"])
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

def solve_sim(key, fun, W, x0, theta,
             tmin, tmax, n_res,
             trans_state, mean_state, var_state,
             trans_obs, mean_obs, var_obs, y_obs, 
             interrogate=rode.interrogate_rodeo):
    r"""
    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Determines number of evaluations between observations; resolution number.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        trans_obs (ndarray(n_obs, n_state)): Transition matrix defining the noisy observations; :math:`C`.
        mean_obs (ndarray(n_obs)): Transition offsets defining the noisy observations.
        var_obs (ndarray(n_obs, n_obs)): Variance matrix defining the noisy observations; :math:`Omega`.
        interrogate (function): Function defining the interrogation method.


    Returns:
        (ndarray(n_steps+1, n_blocks, n_bstate)): Sample solution for :math:`X_t` at times :math:`t \in [a, b]`.

    """
    # Reshaping y_obs to be in blocks 
    n_obs, n_block, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_block, n_bobs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)
    
    # prng keys
    key, *subkeys = jax.random.split(key, num=n_steps*n_block+1)
    subkeys = jnp.reshape(jnp.array(subkeys), newshape=(n_steps, n_block, 2))

    # forward pass
    filt_out = _solve_filter(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_steps=n_steps, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        trans_obs=trans_obs, mean_obs=mean_obs, 
        var_obs=var_obs, y_obs=y_obs,
        interrogate=interrogate
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

        def vmap_fun(b):
            mean_state_sim, var_state_sim = smooth_sim(
                x_state_next=x_state_next[b],
                trans_state=trans_state[b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b]
            )
            return jax.random.multivariate_normal(key[b], mean_state_sim, var_state_sim)

        x_state_curr = jax.vmap(lambda b:
            vmap_fun(b)
        )(jnp.arange(n_block))
        return x_state_curr, x_state_curr
    # initialize
    scan_init = jax.vmap(lambda b:
                         jax.random.multivariate_normal(
                             subkeys[n_steps-1, b], 
                             mean_state_filt[n_steps, b],
                             var_state_filt[n_steps, b]))(jnp.arange(n_block))
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1],
        'key': subkeys[:n_steps-1]
    }
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    x_state_smooth = jnp.concatenate(
        [x0[None], scan_out, scan_init[None]]
    )
    return x_state_smooth

def solve_mv(key, fun, W, x0, theta,
             tmin, tmax, n_res,
             trans_state, mean_state, var_state,
             trans_obs, mean_obs, var_obs, y_obs, 
             interrogate=rode.interrogate_rodeo):
    r"""
    Mean and variance of the stochastic ODE solver using observations.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Determines number of evaluations between observations; resolution number.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        trans_obs (ndarray(n_obs, n_state)): Transition matrix defining the noisy observations; :math:`C`.
        mean_obs (ndarray(n_obs)): Transition offsets defining the noisy observations.
        var_obs (ndarray(n_obs, n_obs)): Variance matrix defining the noisy observations; :math:`Omega`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    # Reshaping y_obs to be in blocks 
    n_obs, n_block, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_block, n_bobs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)

    # get dimensions
    n_block, n_bstate = mean_state.shape
    # forward pass
    filt_out = _solve_filter(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_steps=n_steps, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        trans_obs=trans_obs, mean_obs=mean_obs, 
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
                trans_state=trans_state[b],
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
                     W, x_meas, mean_meas,
                     trans_meas, var_meas):
    r"""
    Perform one update step of the Kalman filter and forecast.

    Args:
        mean_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        x_meas (ndarray(n_meas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mean_meas (ndarray(n_meas)): Transition offsets defining the measure prior; denoted by :math:`d`.
        W (ndarray(n_meas, n_state)): Matrix for getting the derivative; denoted by :math:`W`.
        trans_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; denoted by :math:`W+B`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`\Sigma_n`.

    Returns:
        (tuple):
        - **logdens** (float): The log-likelihood for the observations.
        - **mean_state_filt** (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n}`.
        - **var_state_filt** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
    """
    # kalman forecast
    mean_state_fore, var_state_fore = forecast(
        mean_state_pred = mean_state_pred,
        var_state_pred = var_state_pred,
        W = W,
        mean_meas = mean_meas,
        trans_meas = trans_meas,
        var_meas = var_meas
    )
    logdens = jsp.stats.multivariate_normal.logpdf(x_meas, mean=mean_state_fore, cov=var_state_fore)
    # kalman update
    mean_state_filt, var_state_filt = update(
        mean_state_pred = mean_state_pred,
        var_state_pred = var_state_pred,
        W = W,
        x_meas = x_meas,
        mean_meas = mean_meas,
        trans_meas = trans_meas,
        var_meas = var_meas
    )
    return logdens, mean_state_filt, var_state_filt

def loglikehood(key, fun, W, x0, theta,
                tmin, tmax, n_res,
                trans_state, mean_state, var_state,
                trans_obs, mean_obs, var_obs, y_obs, 
                interrogate):
    r"""
    Compute marginal loglikelihood of pope algorithm.

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
        - **logdens** (float)): Compute the loglikelihood of :math:`p(y|theta, z=0)` or :math:`p(y, z=0|theta)$

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

# --------------------------------------------- Non Gaussian Observations ------------------------------------------------------------------

def _solve_filter_nn(key, fun, W, x0, theta,
                     tmin, tmax, n_steps,
                     trans_state, mean_state, var_state,
                     fun_obs, trans_obs, y_obs, 
                     interrogate):
    r"""
    Forward pass of the pope algorithm.

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
    # Dimensions of block, state, measure and observation variables
    n_block, n_bmeas, n_bstate = W.shape
    n_bobs = y_obs.shape[2]

    # arguments for kalman_filter and kalman_smooth
    mean_state_init = x0
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))
    x_meas = jnp.zeros((n_block, n_bmeas))
    mean_obs = jnp.zeros((n_block, n_bobs))

    # lax.scan setup
    # scan function
    def scan_fun(carry, args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        key, subkey = jax.random.split(carry["key"])
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
            # transform to yhat
            Cmu = jax.vmap(lambda b: trans_obs[b].dot(mean_state_pred[b]))(jnp.arange(n_block))
            gpmu = jax.jacfwd(fun_obs)(Cmu, y_curr)
            gppmu = jax.jacfwd(jax.jacrev(fun_obs))(Cmu, y_curr)
            var_obs = jax.vmap(lambda b: -jnp.linalg.inv(gppmu[b, :, b]))(jnp.arange(n_block))
            y_new = jax.vmap(lambda b: Cmu[b] + var_obs[b].dot(gpmu[b]))(jnp.arange(n_block))
            # stack measure and observation variables
            trans_meas_obs = jnp.concatenate([trans_meas, trans_obs], axis=1)
            mean_meas_obs = jnp.concatenate([mean_meas, mean_obs], axis=1)
            var_meas_obs = jax.vmap(lambda b: jsp.linalg.block_diag(var_meas[b], var_obs[b]))(jnp.arange(n_block))
            x_meas_obs = jnp.concatenate([x_meas, y_new], axis=1)
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

def solve_mv_nn(key, fun, W, x0, theta,
             tmin, tmax, n_res,
             trans_state, mean_state, var_state,
             fun_obs, trans_obs, y_obs, 
             interrogate=rode.interrogate_rodeo):
    r"""
    Mean and variance of the stochastic ODE solver using observations.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_res (int): Determines number of evaluations between observations; resolution number.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        trans_obs (ndarray(n_obs, n_state)): Transition matrix defining the noisy observations; :math:`C`.
        mean_obs (ndarray(n_obs)): Transition offsets defining the noisy observations.
        var_obs (ndarray(n_obs, n_obs)): Variance matrix defining the noisy observations; :math:`Omega`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    # get dimensions
    n_block, n_bstate = mean_state.shape
    # Reshaping y_obs to be in blocks 
    n_obs, n_block, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_block, n_bobs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)

    # forward pass
    filt_out = _solve_filter_nn(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_steps=n_steps, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        fun_obs=fun_obs, trans_obs=trans_obs, y_obs=y_obs,
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
                trans_state=trans_state[b],
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
              trans_state):
    r"""
    Compute the loglikelihood of :math:`p(\mu \mid \hat y, z=0)`.
    
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
                trans_state=trans_state[b]
            )
        )(jnp.arange(n_block))
        mean_state_sim, var_state_sim = jax.vmap(lambda b:
            smooth_sim(
                x_state_next=state_next["mean"][b],
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                trans_state=trans_state[b]
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
    scan_out["var"] = jnp.concatenate(
        [var_state_filt[0][None], scan_out["var"], scan_init["var"][None]]
    )
    return scan_out["mean"], scan_out["var"], scan_out["logx_yhat"]

def _logx_z(uncond_mean, 
            mean_state_filt, var_state_filt,
            mean_state_pred, var_state_pred,
            trans_state):
    r"""
    Compute the loglikelihood of :math:`p(\mu \mid z=0)`.
    
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
                trans_state=trans_state[b]
            )
        )(jnp.arange(n_block))
        logx_z += jnp.sum(
            jax.vmap(lambda b:
                     multivariate_normal_logpdf(uncond_curr[b], mean=mean_state_sim[b], cov=var_state_sim[b])
                    )(jnp.arange(n_block))
        )
        state_curr = {
            "logx_z":  logx_z,
            "var": var_state_sim
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
        "var": var_state_filt[-1]
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
    
    return scan_out["logx_z"], scan_out["var"]

def loglikehood_nn(key, fun, W, x0, theta,
                tmin, tmax, n_res,
                trans_state, mean_state, var_state,
                fun_obs, trans_obs, y_obs, 
                interrogate=rode.interrogate_rodeo):
    r"""
    Compute marginal loglikelihood of pope algorithm.

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
        - **logdens** (float)): Compute the loglikelihood of :math:`p(y|theta, z=0)` or :math:`p(y, z=0|theta)`

    """
    # Reshaping y_obs to be in blocks 
    n_obs, n_block, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_block, n_bobs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)

    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape

    # forward pass
    filt_out = _solve_filter_nn(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_steps=n_steps, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        fun_obs=fun_obs, trans_obs=trans_obs, y_obs=y_obs,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # logp(x | hat y, z)
    mean_state_smooth, var_state_smooth, logx_yhat = _logx_yhat(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        trans_state=trans_state
    )

    # logp(y | x)
    def vmap_fun(t):
        Cx = jax.vmap(lambda b: trans_obs[b].dot(mean_state_smooth[t][b]))(jnp.arange(n_block))
        return fun_obs(Cx, y_obs[t])
    logy_x = jnp.sum(jax.vmap(vmap_fun)(jnp.arange(0, n_steps+1, n_res)))

    # logp(x | z)
    # first do forward pass without obs
    filt_out = rode._solve_filter(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_steps=n_steps, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    logx_z, var_state_smooth2 = _logx_z(
        uncond_mean=mean_state_smooth,
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        trans_state=trans_state
    )

    return logy_x + logx_z[0] - logx_yhat[0]
    # return scan_out
