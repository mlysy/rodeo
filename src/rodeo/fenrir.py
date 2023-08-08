r"""
This module implements the Fenrir algorithm as described in Tronarp et al 2022 for computing the approximate likelihood of :math:`p(y_{0:M} \mid z_{0:N})`.

The forward pass model is

.. math::

    x_0 = v

    X_n = c_n + Q_n X_{n-1} + R_n^{1/2} \epsilon_n

    z_n = W_n X_n - f(X_n, t_n) + V_n^{1/2} \eta_n.

We assume that :math:`c_n = c, Q_n = Q, R_n = R`, and :math:`W_n = W` for all :math:`n`. Using the Kalman filtering recursions, the above model can be simulated via the reverse pass model

.. math::

    X_N \sim N(b_N, C_N)

    X_n = A_n X_{n+1} + b_n + C_n^{1/2} \epsilon_n.
    
Fenrir combines the observations

.. math::

    y_m = D_m X_m + \Omega^{1/2} \epsilon_m,

with the reverse pass model to condition on data.
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.kalmantv import *
from rodeo.utils import multivariate_normal_logpdf

# use linearizations first then observations
def forward(key, fun, W, x0, theta,
            tmin, tmax, n_steps,
            trans_state, var_state,
            interrogate):
    r"""
    Forward pass of the Fenrir algorithm.

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
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
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

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    mean_state = jnp.zeros((n_block, n_bstate))
    mean_state_init = x0
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # lax.scan setup
    # scan function
    def scan_fun(carry, t):
        mean_state_filt, var_state_filt = carry["state_filt"]
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
            key=key,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        W_meas = W + trans_meas
        # kalman update
        mean_state_next, var_state_next = jax.vmap(lambda b:
            update(
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                x_meas=x_meas[b],
                mean_meas=mean_meas[b],
                trans_meas=W_meas[b],
                var_meas=var_meas[b]
            )
        )(jnp.arange(n_block))
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

def backward_param(mean_state_filt, var_state_filt, 
                   mean_state_pred, var_state_pred,
                   trans_state):
    r"""
    Compute the backward markov chain parameters.

    Args:
        mean_state_pred (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        var_state_pred (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        mean_state_filt (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        var_state_filt (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        
    Returns:
        (tuple):
        - **trans_state_cond** (ndarray(n_steps, n_block, n_bstate, n_bstate)): Transition of smooth conditional at time t given observations from times [0...T]; :math:`A_{n|N}`.
        - **mean_state_cond** (ndarray(n_steps+1, n_block, n_bstate)): Offset of smooth conditional at time t given observations from times [0...T]; :math:`b_{n|N}`.
        - **var_state_cond** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance of smooth conditional at time t given observations from times [0...T]; :math:`V_{n|N}`.

    """
    # Terminal Point
    n_tot, n_block, _ = mean_state_filt.shape
    mean_state_end = mean_state_filt[n_tot-1]
    var_state_end = var_state_filt[n_tot-1]

    # backward pass
    # vmap setup
    def vmap_fun(smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        
        trans_state_cond, mean_state_cond, var_state_cond = jax.vmap(lambda b:
            smooth_cond(
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                trans_state=trans_state[b]
            )
        )(jnp.arange(n_block))
        return trans_state_cond, mean_state_cond, var_state_cond

    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[:n_tot-1],
        'var_state_filt': var_state_filt[:n_tot-1],
        'mean_state_pred': mean_state_pred[1:n_tot],
        'var_state_pred': var_state_pred[1:n_tot],
    }
    trans_state_cond, mean_state_cond, var_state_cond = jax.vmap(vmap_fun)(scan_kwargs)
    mean_state_cond = jnp.concatenate([mean_state_cond, mean_state_end[None]])
    var_state_cond = jnp.concatenate([var_state_cond, var_state_end[None]])
    return trans_state_cond, mean_state_cond, var_state_cond

def backward(n_res, trans_state, mean_state, var_state,
             trans_obs, var_obs, y_obs):
    
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
    n_obs, n_block, n_bobs, n_bstate = trans_obs.shape
    n_steps = (n_obs-1)*n_res
    trans_state_end = jnp.zeros(trans_state.shape[1:])
    trans_state = jnp.concatenate([trans_state, trans_state_end[None]])

    # offset of obs is assumed to be 0
    mean_obs = jnp.zeros((n_block, n_bobs))

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
            trans_curr = trans_obs[i]
            # kalman forecast
            mean_state_fore, var_state_fore = jax.vmap(lambda b:
                forecast(
                    mean_state_pred = mean_state_pred[b],
                    var_state_pred = var_state_pred[b],
                    mean_meas = mean_obs[b],
                    trans_meas = trans_curr[b],
                    var_meas = var_obs[b]
                    )
            )(jnp.arange(n_block))
            # logdensity of forecast
            # logp = 0.0
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
                    trans_meas=trans_curr[b],
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
    
def fenrir(key, fun, W, x0, theta, tmin, tmax, n_res,
           trans_state, var_state,
           trans_obs, var_obs, y_obs,
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
        n_res=n_res, trans_state=trans_state_cond, 
        mean_state=mean_state_cond, var_state=var_state_cond, 
        trans_obs=trans_obs, var_obs=var_obs, y_obs=y_obs
    )

    return logdens

def _smooth_mv(trans_state, state_par):
    r"""
    Smoothing pass of the Fenrir algorithm used to compute solution posterior.

    Args:
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        state_par (dict): Dictionary containing the mean and variance matrices of the predicted and updated steps of the Kalman filter.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    mean_state_pred, var_state_pred = state_par["state_pred"]
    mean_state_filt, var_state_filt = state_par["state_filt"]
    n_tot, n_block, n_bstate = mean_state_pred.shape
    # smooth pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        trans_state = smooth_kwargs['trans_state']
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
        "mean": mean_state_filt[1],
        "var": var_state_filt[1]
    }
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[2:],
        'var_state_filt': var_state_filt[2:],
        'mean_state_pred': mean_state_pred[1:n_tot-1],
        'var_state_pred': var_state_pred[1:n_tot-1],
        'trans_state': trans_state[1:n_tot]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs)

    # append initial values to front and back
    mean_state_smooth = jnp.concatenate(
        [mean_state_filt[0:2], scan_out["mean"]]
    )
    var_state_smooth = jnp.concatenate(
        [var_state_filt[0:2], scan_out["var"]]
    )
    return mean_state_smooth, var_state_smooth

def fenrir_mv(key, fun, W, x0, theta, tmin, tmax, n_res,
              trans_state, var_state,
              trans_obs, var_obs, y_obs, interrogate):
    
    r"""
    Fenrir algorithm to compute the mean and variance of :math:`X_{0:N}`.

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
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

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
    _, state_par = backward(
        n_res=n_res, trans_state=trans_state_cond, 
        mean_state=mean_state_cond, var_state=var_state_cond, 
        trans_obs=trans_obs, var_obs=var_obs, y_obs=y_obs
    )
    mean_state_smooth, var_state_smooth = _smooth_mv(trans_state_cond, state_par)
    return mean_state_smooth, var_state_smooth
