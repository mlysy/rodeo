r"""
This module implements the Fenrir algorithm as described in Tronarp et al 2022 for computing the approximate likelihood of :math:`p(Y_{0:M} \mid Z_{1:N})`.

The forward pass model is

.. math::

    x_0 = v

    X_n = c_n + Q_n X_{n-1} + R_n^{1/2} \epsilon_n

    Z_n = W_n X_n - f(X_n, t_n) + V_n^{1/2} \eta_n.

We assume that :math:`c_n = 0, Q_n = Q, R_n = R`, and :math:`W_n = W` for all :math:`n`. Using the Kalman filtering recursions, the above model can be simulated via the reverse pass model

.. math::

    X_N \sim \operatorname{Normal}(b_N, C_N)

    X_n = A_n X_{n+1} + b_n + C_n^{1/2} \epsilon_n.
    
Fenrir combines the observations

.. math::

    Y_m = D_m X_m + \Omega^{1/2}_m \eta_m,

with the reverse pass model to condition on data. Here :math:`\epsilon_n, \eta_m` are standard normals.
"""
import jax
import jax.numpy as jnp
from kalmantv import standard
from rodeo.solve import _solve_filter
from rodeo.utils import multivariate_normal_logpdf

# --- helper functions --------------------------------------------------------


def _forecast_update(mean_state_pred, var_state_pred,
                     x_meas, mean_meas,
                     wgt_meas, var_meas,
                     kalman_funs):
    r"""
    Perform one update step of the Kalman filter and forecast.

    Args:
        mean_state_pred (ndarray(n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_block, n_bstate, n_sbtate)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        x_meas (ndarray(n_block, n_bmeas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mean_meas (ndarray(n_block, n_bmeas)): Transition offsets defining the measure prior.
        wgt_meas (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior.
        var_meas (ndarray(n_block, n_bmeas, n_bmeas)): Variance matrix defining the measure prior.
        kalman_funs (object): An object that contains the Kalman filtering functions: predict, update and smooth.

    Returns:
        (tuple):
        - **logdens** (float): The loglikelihood for the observations.
        - **mean_state_filt** (ndarray(n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n}`.
        - **var_state_filt** (ndarray(n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
    """
    # kalman forecast
    mean_state_fore, var_state_fore = kalman_funs.forecast(
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        mean_meas=mean_meas,
        wgt_meas=wgt_meas,
        var_meas=var_meas
    )
    logdens = multivariate_normal_logpdf(
        x_meas, mean=mean_state_fore, cov=var_state_fore)
    # kalman update
    mean_state_filt, var_state_filt = kalman_funs.update(
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        x_meas=x_meas,
        mean_meas=mean_meas,
        wgt_meas=wgt_meas,
        var_meas=var_meas
    )
    return logdens, mean_state_filt, var_state_filt

# --- loglikelihood -----------------------------------------------------------


def _backward(mean_state_filt, var_state_filt,
              mean_state_pred, var_state_pred,
              prior_weight, prior_var,
              t_min, t_max, n_steps,
              obs_data, obs_times,
              obs_weight, obs_var,
              kalman_funs):
    r"""
    Compute the backward Markov chain parameters and forward pass but backwards in time.

    Args:
        mean_state_pred (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        mean_state_filt (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n}`.
        var_state_filt (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
        prior_weight (ndarray(n_block, n_bstate, n_bstate)): Weight matrix defining the solution prior; :math:`Q`.
        prior_var (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        t_min (float): First time point of the time interval to be evaluated; :math:`t_0`.
        t_max (float): Last time point of the time interval to be evaluated; :math:`t_N`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        obs_data (ndarray(n_obs, n_blocks, n_bobs)): Observed data; :math:`y_{0:M}`.
        obs_times (ndarray(n_obs)): Observation time; :math:`0, \ldots, M`.
        obs_weight (ndarray(n_obs, n_blocks, n_bobs, n_bstate)): Weight matrix in the observation model; :math:`D_{0:M}`.
        obs_var (ndarry(n_obs, n_blocks, n_bobs, n_bobs)): Variance matrix in the observation model; :math:`\Omega_{0:M}`
        kalman_funs (object): An object that contains the Kalman filtering functions: predict, update and smooth.

    Returns:
        (float) : The logdensity of :math:`p(y_{0:M} \mid Z_{1:N})`.

    """
    # get dimensions
    n_obs, n_block, n_bobs, n_bstate = obs_weight.shape
    # insert observations on solver time grid
    sim_times = jnp.linspace(t_min, t_max, n_steps + 1)
    obs_ind = jnp.searchsorted(sim_times, obs_times)
    # offset of obs is assumed to be 0
    obs_mean = jnp.zeros((n_block, n_bobs))
    # forecast function without kalman_funs
    forecast_update = lambda mean_state_pred, var_state_pred,\
                             x_meas, mean_meas, wgt_meas, var_meas\
                             : _forecast_update(mean_state_pred, var_state_pred,
                                                x_meas, mean_meas,
                                                wgt_meas, var_meas,
                                                kalman_funs)

    def scan_fun(carry, forward_states):
        # Kalman filter backwards in time
        bmean_state_filt, bvar_state_filt = carry["state_filt"] 
        # Kalman filter estimates from forward
        mean_state_filt, var_state_filt = forward_states["state_filt"]
        mean_state_pred, var_state_pred = forward_states["state_pred"]

        logdens = carry["logdens"]
        i = carry["i"]
        t = forward_states["t"] # t_n
        # get Markov params
        wgt_state_back, mean_state_back, var_state_back = jax.vmap(kalman_funs.smooth_cond)(
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                wgt_state=prior_weight,
                var_state=prior_var
        )
        # kalman predict
        bmean_state_pred, bvar_state_pred = jax.vmap(kalman_funs.predict)(
                mean_state_past=bmean_state_filt,
                var_state_past=bvar_state_filt,
                mean_state=mean_state_back,
                wgt_state=wgt_state_back,
                var_state=var_state_back
        )

        # not t time point of observation
        def _no_obs():
            return bmean_state_pred, bvar_state_pred, 0.0, i

        
        # at time point of observation
        def _obs():
            # kalman forecast and update
            logp, bmean_state_next, bvar_state_next = jax.vmap(forecast_update)(
                    mean_state_pred=bmean_state_pred,
                    var_state_pred=bvar_state_pred,
                    x_meas=obs_data[i],
                    mean_meas=obs_mean,
                    wgt_meas=obs_weight[i],
                    var_meas=obs_var[i]
            )
            return bmean_state_next, bvar_state_next, jnp.sum(logp), i-1

        bmean_state_filt, bvar_state_filt, logp, i = jax.lax.cond(
            obs_ind[i] == t, _obs, _no_obs)
        logdens += logp

        # output
        carry = {
            "state_filt": (bmean_state_filt, bvar_state_filt),
            "logdens": logdens,
            "i": i
        }
        stack = {
            "state_pred": (bmean_state_pred, bvar_state_pred),
            "state_filt": (bmean_state_filt, bvar_state_filt),
            "wgt_state": wgt_state_back,
            "var_state": var_state_back 
        }
        return carry, stack

    # terminal point update
    mean_state_term = mean_state_filt[n_steps]
    var_state_term = var_state_filt[n_steps]
    logdens = 0.0
    i = n_obs - 1
    # no observations
    def _no_obs():
        # no need to update
        return mean_state_term, var_state_term, 0.0, i

    # observation
    def _obs():
        # kalman forecast and update
        logp, bmean_state_next, bvar_state_next = jax.vmap(forecast_update)(
                mean_state_pred=mean_state_term,
                var_state_pred=var_state_term,
                x_meas=obs_data[i],
                mean_meas=obs_mean,
                wgt_meas=obs_weight[i],
                var_meas=obs_var[i]
        )
        return bmean_state_next, bvar_state_next, jnp.sum(logp), i-1

    bmean_state_filt, bvar_state_filt, logp, i = jax.lax.cond(
        obs_ind[i] >= n_steps, _obs, _no_obs)
    logdens += logp

    # start at N 
    scan_init = {
        "state_filt": (bmean_state_filt, bvar_state_filt),
        "logdens": logdens,
        "i": i
    }
    forward_states_init = {
        "state_pred": (mean_state_pred[1:n_steps+1], var_state_pred[1:n_steps+1]),
        "state_filt": (mean_state_filt[:n_steps], var_state_filt[:n_steps]),
        "t": jnp.arange(n_steps)
    }

    scan_out, scan_out2 = jax.lax.scan(
        scan_fun, scan_init, forward_states_init, reverse=True)

    # append initial values to back
    mean_scan_pred, var_scan_pred = scan_out2["state_pred"]
    mean_scan_filt, var_scan_filt = scan_out2["state_filt"]
    mean_state_pred = jnp.concatenate(
        [mean_scan_pred, mean_state_term[None]]
    )
    var_state_pred = jnp.concatenate(
        [var_scan_pred, var_state_term[None]]
    )
    mean_state_filt = jnp.concatenate(
        [mean_scan_filt, bmean_state_filt[None]]
    )
    var_state_filt = jnp.concatenate(
        [var_scan_filt, bvar_state_filt[None]]
    )
    # repack
    scan_out2 = {
        "state_pred": (mean_state_pred, var_state_pred),
        "state_filt": (mean_state_filt, var_state_filt),
        "wgt_state": scan_out2["wgt_state"],
        "var_state": scan_out2["var_state"]
    }
    return scan_out["logdens"], scan_out2

def fenrir(key, ode_fun, ode_weight, ode_init,
           t_min, t_max, n_steps,
           interrogate,
           prior_weight, prior_var,
           obs_data, obs_times, obs_weight, obs_var,
           kalman_funs=standard, **params):
    r"""
    Fenrir algorithm to compute the approximate loglikelihood of :math:`p(Y_{0:M} \mid Z_{1:N})`.

    Args:
        key (PRNGKey): PRNG key.
        ode_fun (Callable): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        ode_weight (ndarray(n_block, n_bmeas, n_bstate)): Weight matrix defining the measure prior; :math:`W`.
        ode_init (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        t_min (float): First time point of the time interval to be evaluated; :math:`a`.
        t_max (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        interrogate (Callable): Function defining the interrogation method.
        prior_weight (ndarray(n_block, n_bstate, n_bstate)): Weight matrix defining the solution prior; :math:`Q`.
        prior_var (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        obs_data (ndarray(n_obs, n_blocks, n_bobs)): Observed data; :math:`y_{0:M}`.
        obs_times (ndarray(n_obs)): Observation time; :math:`0, \ldots, M`.
        obs_weight (ndarray(n_obs, n_blocks, n_bobs, n_bstate)): Weight matrix in the observation model; :math:`D_{0:M}`.
        obs_var (ndarry(n_obs, n_blocks, n_bobs, n_bobs)): Variance matrix in the observation model; :math:`\Omega_{0:M}`
        kalman_funs (object): An object that contains the Kalman filtering functions: predict, update and smooth.
        params (kwargs): Optional model parameters.

    Returns:
        (float) : The loglikelihood of :math:`p(y_{0:M} \mid Z_{1:N})`.

    """

    # forward pass
    filt_out = _solve_filter(
        key=key,
        ode_fun=ode_fun, ode_weight=ode_weight, ode_init=ode_init,
        t_min=t_min, t_max=t_max, n_steps=n_steps,
        interrogate=interrogate,
        prior_weight=prior_weight, prior_var=prior_var,
        kalman_funs=kalman_funs, **params
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    logdens, _ = _backward(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        prior_weight=prior_weight,
        prior_var=prior_var,
        t_min=t_min, t_max=t_max, n_steps=n_steps,
        obs_data=obs_data, obs_times=obs_times,
        obs_weight=obs_weight, obs_var=obs_var,
        kalman_funs=kalman_funs
    )

    return logdens

# --- ODE solver --------------------------------------------------------------


def _smooth_mv(state_par, kalman_funs):
    r"""
    Smoothing pass of the Fenrir algorithm used to compute solution posterior.

    Args:
        state_par (dict): Dictionary containing the weight, mean and variance matrices of the predicted and updated steps of the Kalman filter.
        kalman_funs (object): An object that contains the Kalman filtering functions: predict, update and smooth.


    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    mean_state_pred, var_state_pred = state_par["state_pred"]
    mean_state_filt, var_state_filt = state_par["state_filt"]
    wgt_state = state_par["wgt_state"]
    var_state = state_par["var_state"]
    n_tot = mean_state_pred.shape[0]
    # smooth pass
    # lax.scan setup

    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        wgt_state = smooth_kwargs['wgt_state']
        var_state = smooth_kwargs['var_state']
        mean_state_curr, var_state_curr = jax.vmap(kalman_funs.smooth_mv)(
                mean_state_next=state_next["mean"],
                var_state_next=state_next["var"],
                wgt_state=wgt_state,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                var_state=var_state
        )
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
        'wgt_state': wgt_state[1:n_tot],
        'var_state': var_state[1:n_tot]
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
             kalman_funs=standard, **params):
    r"""
    Fenrir algorithm to compute the mean and variance of :math:`p(X_{0:N} \mid Z_{1:N}, Y_{0:M})`. Same arguments as :func:`~fenrir.fenrir`.

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
        kalman_funs=kalman_funs, **params
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    _, state_par = _backward(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        prior_weight=prior_weight,
        prior_var=prior_var,
        t_min=t_min, t_max=t_max, n_steps=n_steps,
        obs_data=obs_data, obs_times=obs_times,
        obs_weight=obs_weight, obs_var=obs_var,
        kalman_funs=kalman_funs
    )

    mean_state_smooth, var_state_smooth = _smooth_mv(state_par, kalman_funs)
    return mean_state_smooth, var_state_smooth
