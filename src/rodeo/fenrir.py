r"""
This module implements the Fenrir algorithm as described in Tronarp et al 2022 for computing the approximate likelihood of :math:`p(y_{0:M} \mid z_{0:N})`.

The forward pass model is

.. math::

    x_0 = v

    X_n = c_n + Q_n X_{n-1} + R_n^{1/2} \epsilon_n

    Z_n = W_n X_n - f(X_n, t_n) + V_n^{1/2} \eta_n.

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
from rodeo.ode import _solve_filter
from rodeo.utils import multivariate_normal_logpdf

def backward_param(mean_state_filt, var_state_filt, 
                   mean_state_pred, var_state_pred,
                   prior_weight):
    r"""
    Compute the backward Markov chain parameters.

    Args:
        mean_state_pred (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        var_state_pred (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        mean_state_filt (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        var_state_filt (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        prior_weight (ndarray(n_block, n_bstate, n_bstate)): Weight matrix defining the solution prior; :math:`Q`.
        
    Returns:
        (tuple):
        - **wgt_state_cond** (ndarray(n_steps, n_block, n_bstate, n_bstate)): Weight of smooth conditional at time t given observations from times [0...T]; :math:`A_{n|N}`.
        - **mean_state_cond** (ndarray(n_steps+1, n_block, n_bstate)): Offset of smooth conditional at time t given observations from times [0...T]; :math:`b_{n|N}`.
        - **var_state_cond** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance of smooth conditional at time t given observations from times [0...T]; :math:`C_{n|N}`.

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
        
        wgt_state_cond, mean_state_cond, var_state_cond = jax.vmap(lambda b:
            smooth_cond(
                mean_state_filt=mean_state_filt[b],
                var_state_filt=var_state_filt[b],
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                wgt_state=prior_weight[b]
            )
        )(jnp.arange(n_block))
        return wgt_state_cond, mean_state_cond, var_state_cond

    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[:n_tot-1],
        'var_state_filt': var_state_filt[:n_tot-1],
        'mean_state_pred': mean_state_pred[1:n_tot],
        'var_state_pred': var_state_pred[1:n_tot],
    }
    wgt_state_cond, mean_state_cond, var_state_cond = jax.vmap(vmap_fun)(scan_kwargs)
    mean_state_cond = jnp.concatenate([mean_state_cond, mean_state_end[None]])
    var_state_cond = jnp.concatenate([var_state_cond, var_state_end[None]])
    return wgt_state_cond, mean_state_cond, var_state_cond

def backward(t_min, t_max, n_steps,
             wgt_state, mean_state, var_state,
             obs_data, obs_times,
             obs_weight, obs_var):
    
    r"""
    Backward pass of Fenrir algorithm where observations are used.

    Args:
        t_min (float): First time point of the time interval to be evaluated; :math:`a`.
        t_max (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        wgt_state (ndarray(n_steps, n_block, n_bstate, n_bstate)): Weight matrix defining the solution prior; :math:`A_{0:N}`.
        mean_state (ndarray(n_steps+1, n_block, n_bstate)): Offsets defining the solution prior; :math:`b_{0:N}`.
        var_state (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`C_{0:N}`.
        obs_data (ndarray(n_obs, n_blocks, n_bobs)): Observed data; :math:`y_{0:M}`.
        obs_times (ndarray(n_obs)): Observation time; :math:`0, \ldots, M`.
        obs_weight (ndarray(n_obs, n_blocks, n_bobs, n_bstate)): Weight matrix in the observation model; :math:`D_{0:M}`.
        obs_var (ndarry(n_obs, n_blocks, n_bobs, n_bobs)): Variance matrix in the observation model; :math:`\Omega_{0:M}`

    Returns:
        (float) : The logdensity of :math:`p(y_{0:M} \mid z_{0:N})`.

    """
    # Add point to beginning of state variable for simpler loop
    n_obs, n_block, n_bobs, n_bstate = obs_weight.shape
    # n_steps = (n_obs-1)*n_res
    wgt_state_end = jnp.zeros(wgt_state.shape[1:])
    wgt_state = jnp.concatenate([wgt_state, wgt_state_end[None]])

    # offset of obs is assumed to be 0
    obs_mean = jnp.zeros((n_block, n_bobs))

    # reverse pass
    def scan_fun(carry, t):
        mean_state_filt, var_state_filt = carry["state_filt"]
        logdens = carry["logdens"]
        i = carry["i"]
        ode_time = t_min + (t_max-t_min)*t/n_steps

        # kalman predict
        mean_state_pred, var_state_pred = jax.vmap(lambda b:
            predict(
                mean_state_past=mean_state_filt[b],
                var_state_past=var_state_filt[b],
                mean_state=mean_state[t, b],
                wgt_state=wgt_state[t, b],
                var_state=var_state[t, b]
            )
        )(jnp.arange(n_block))
        # not t time point of observation
        def _no_obs():
            return mean_state_pred, var_state_pred, 0.0, i

        # at time point of observation
        def _obs():
            # kalman forecast
            mean_state_fore, var_state_fore = jax.vmap(lambda b:
                forecast(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    mean_meas=obs_mean[b],
                    wgt_meas=obs_weight[i, b],
                    var_meas=obs_var[i, b]
                )
            )(jnp.arange(n_block))
            # logdensity of forecast
            logp = jnp.sum(jax.vmap(lambda b:
                multivariate_normal_logpdf(obs_data[i, b], mean=mean_state_fore[b], cov=var_state_fore[b])
            )(jnp.arange(n_block)))
            # kalman update
            mean_state_filt, var_state_filt = jax.vmap(lambda b:
                update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    x_meas=obs_data[i, b],
                    mean_meas=obs_mean[b],
                    wgt_meas=obs_weight[i, b],
                    var_meas=obs_var[i, b]
                )
            )(jnp.arange(n_block))
            return mean_state_filt, var_state_filt, logp, i-1

        mean_state_filt, var_state_filt, logp, i = jax.lax.cond(ode_time == obs_times[i], _obs, _no_obs)
        logdens += logp

        # output
        carry = {
            "state_filt": (mean_state_filt, var_state_filt),
            "logdens" : logdens,
            "i" : i
        }
        stack = {
            "state_pred": (mean_state_pred, var_state_pred),
            "state_filt": (mean_state_filt, var_state_filt)
        }
        return carry, stack

    # start at N+1 assuming 0 mean and variance
    scan_init = {
        "state_filt" : (jnp.zeros((n_block, n_bstate,)), jnp.zeros((n_block, n_bstate, n_bstate))),
        "logdens" : 0.0,
        "i": n_obs-1
    }

    scan_out, scan_out2 = jax.lax.scan(scan_fun, scan_init, jnp.arange(n_steps+1), reverse=True)
    return scan_out["logdens"], scan_out2
    
def fenrir(key, ode_fun, ode_weight, ode_init, 
           t_min, t_max, n_steps,
           interrogate,
           prior_weight, prior_var,
           obs_data, obs_times, obs_weight, obs_var,
           **params):
    
    r"""
    Fenrir algorithm to compute the approximate loglikelihood of :math:`p(y_{0:M} \mid z_{0:N})`.

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
        obs_data (ndarray(n_obs, n_blocks, n_bobs)): Observed data; :math:`y_{0:M}`.
        obs_times (ndarray(n_obs)): Observation time; :math:`0, \ldots, M`.
        obs_weight (ndarray(n_obs, n_blocks, n_bobs, n_bstate)): Weight matrix in the observation model; :math:`D_{0:M}`.
        obs_var (ndarry(n_obs, n_blocks, n_bobs, n_bobs)): Variance matrix in the observation model; :math:`\Omega_{0:M}`
        params (kwargs): Optional model parameters.

    Returns:
        (float) : The loglikelihood of :math:`p(y_{0:M} \mid z_{0:N})`.

    """

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
    wgt_state_cond, mean_state_cond, var_state_cond = backward_param(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        prior_weight=prior_weight
    )

    # reverse pass
    logdens, _ = backward(
        t_min=t_min, t_max=t_max, n_steps=n_steps,
        wgt_state=wgt_state_cond, mean_state=mean_state_cond, 
        var_state=var_state_cond, 
        obs_data=obs_data, obs_times=obs_times,
        obs_weight=obs_weight, obs_var=obs_var
    )

    return logdens

def _smooth_mv(wgt_state, state_par):
    r"""
    Smoothing pass of the Fenrir algorithm used to compute solution posterior.

    Args:
        wgt_state (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Weight matrix defining the solution prior; :math:`A_{0:N}`.
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
        wgt_state = smooth_kwargs['wgt_state']
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
        "mean": mean_state_filt[1],
        "var": var_state_filt[1]
    }
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[2:],
        'var_state_filt': var_state_filt[2:],
        'mean_state_pred': mean_state_pred[1:n_tot-1],
        'var_state_pred': var_state_pred[1:n_tot-1],
        'wgt_state': wgt_state[1:n_tot]
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

def solve_mv(key, ode_fun, ode_weight, ode_init, 
              t_min, t_max, n_steps,
              interrogate,
              prior_weight, prior_var,
              obs_data, obs_times, obs_weight, obs_var,
              **params):
    
    r"""
    Fenrir algorithm to compute the mean and variance of :math:`p(X_{0:N} \mid \z`. Same arguments as :func:`~fenrir.fenrir`.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """

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
    wgt_state_cond, mean_state_cond, var_state_cond = backward_param(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        prior_weight=prior_weight
    )

    # reverse pass
    _, state_par = backward(
        t_min=t_min, t_max=t_max, n_steps=n_steps,
        wgt_state=wgt_state_cond, mean_state=mean_state_cond, 
        var_state=var_state_cond, 
        obs_data=obs_data, obs_times=obs_times,
        obs_weight=obs_weight, obs_var=obs_var
    )

    mean_state_smooth, var_state_smooth = _smooth_mv(wgt_state_cond, state_par)
    return mean_state_smooth, var_state_smooth
