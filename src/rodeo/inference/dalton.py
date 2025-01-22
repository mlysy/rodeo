r"""
This module implements the DALTON solver which gives an approximate likelihood of :math:`p(Y_{0:M} \mid Z_{1:N})`.

The model is

.. math::

    x_0 = v

    X_n = c_n + Q_n X_{n-1} + R_n^{1/2} \epsilon_n

    Z_n = W_n X_n - f(X_n, t_n) + V_n^{1/2} \eta_n.
    
    Y_m = g(X_m, \phi_m)

where :math:`g` is a general distribution function. In the case that :math:`g` is Gaussian, use :func:`~dalton.loglikehood` for a better approximation. In other cases, use :func:`~dalton.loglikehood_nn`. We assume that :math:`c_n = 0, Q_n = Q, R_n = R`, and :math:`W_n = W` for all :math:`n`.

In the Gaussian case, we assume the observation model is

.. math::

    Y_m = D_m X_m + \Omega^{1/2}_m \epsilon_m.

We assume that the :math:`M \leq N`, so that the observation step size is larger than that of the evaluation step size.

"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from kalmantv import standard
from rodeo.inference.fenrir import _forecast_update
from rodeo.utils import multivariate_normal_logpdf
from rodeo.solve import _solve_filter as _solve_filter_ode

# --- loglikelihood -----------------------------------------------------------


def dalton(key, ode_fun, ode_weight, ode_init, 
           t_min, t_max, n_steps,
           interrogate,
           prior_weight, prior_var,
           obs_data, obs_times, obs_weight, obs_var,
           kalman_funs=standard, **params):
    r"""
    Compute marginal loglikelihood of DALTON algorithm for Gaussian observations; :math:`p(Y_{0:M} \mid Z_{1:N})`.

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
        obs_data (ndarray(n_obs, n_blocks, n_bobs)): Observed data; :math:`Y_{0:M}`.
        obs_times (ndarray(n_obs)): Observation time; :math:`0, \ldots, M`.
        obs_weight (ndarray(n_obs, n_blocks, n_bobs, n_bstate)): Weight matrix in the observation model; :math:`D_{0:M}`.
        obs_var (ndarry(n_obs, n_blocks, n_bobs, n_bobs)): Variance matrix in the observation model; :math:`\Omega_{0:M}`
        kalman_funs (object): An object that contains the Kalman filtering functions: predict, update and smooth.
        params (kwargs): Optional model parameters.

    Returns:
        (float): Loglikelihood of :math:`p(Y_{0:M} \mid Z_{1:N})`.

    """
    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = ode_weight.shape
    # Dimension of observation
    n_bobs = obs_weight.shape[2]

    # insert observations on solver time grid
    sim_times = jnp.linspace(t_min, t_max, n_steps + 1)
    obs_ind = jnp.searchsorted(sim_times, obs_times)

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    obs_mean = jnp.zeros((n_block, n_bobs))
    mean_state = jnp.zeros((n_block, n_bstate))
    mean_state_init = ode_init
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # forecast function without kalman_funs
    forecast_update = lambda mean_state_pred, var_state_pred,\
                             x_meas, mean_meas, wgt_meas, var_meas\
                             : _forecast_update(mean_state_pred, var_state_pred,
                                                x_meas, mean_meas,
                                                wgt_meas, var_meas,
                                                kalman_funs)

    # compute p(Z_{1:N}, Y_{0:M})
    def scan(carry, filter_kwargs):
        mean_state_filt_zy, var_state_filt_zy = carry["state_filt_joint"]
        mean_state_filt_z, var_state_filt_z = carry["state_filt_marg"]
        logdens_zy = carry["logdens_joint"]
        logdens_z = carry["logdens_marg"]
        t = filter_kwargs["t"]
        keys = filter_kwargs["key"]
        i = carry["i"]
        ode_time = t_min + (t_max-t_min)*(t+1)/n_steps
        
        # compute joint logpdf
        mean_state_pred_zy, var_state_pred_zy = jax.vmap(kalman_funs.predict)(
                mean_state_past=mean_state_filt_zy,
                var_state_past=var_state_filt_zy,
                mean_state=mean_state,
                wgt_state=prior_weight,
                var_state=prior_var
        )
        # compute meas parameters
        wgt_meas, mean_meas, var_meas = interrogate(
            key=keys[0],
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=ode_time,
            mean_state_pred=mean_state_pred_zy,
            var_state_pred=var_state_pred_zy,
            **params
        )
        W_meas = ode_weight + wgt_meas

        # both z and y are observed
        def zy_update():
            wgt_meas_obs = jnp.concatenate([W_meas, obs_weight[i]], axis=1)
            mean_meas_obs = jnp.concatenate([mean_meas, obs_mean], axis=1)
            var_meas_obs = jax.vmap(jsp.linalg.block_diag)(var_meas, obs_var[i])
            x_meas_obs = jnp.concatenate([x_meas, obs_data[i]], axis=1)
            logp, mean_state_next, var_state_next = jax.vmap(forecast_update)(
                    mean_state_pred=mean_state_pred_zy,
                    var_state_pred=var_state_pred_zy,
                    x_meas=x_meas_obs,
                    mean_meas=mean_meas_obs,
                    wgt_meas=wgt_meas_obs,
                    var_meas=var_meas_obs
            )                     
            return mean_state_next, var_state_next, jnp.sum(logp), i+1

        # only z is observed
        def z_update():
            logp, mean_state_next, var_state_next = jax.vmap(forecast_update)(
                    mean_state_pred=mean_state_pred_zy,
                    var_state_pred=var_state_pred_zy,
                    x_meas=x_meas,
                    mean_meas=mean_meas,
                    wgt_meas=W_meas,
                    var_meas=var_meas                                               
            )
            return mean_state_next, var_state_next, jnp.sum(logp), i

        mean_state_next_zy, var_state_next_zy, logp, i = jax.lax.cond(t+1 == obs_ind[i], zy_update, z_update)
        logdens_zy += logp
        

        # compute marginal logpdf
        mean_state_pred_z, var_state_pred_z = jax.vmap(kalman_funs.predict)(
                mean_state_past=mean_state_filt_z,
                var_state_past=var_state_filt_z,
                mean_state=mean_state,
                wgt_state=prior_weight,
                var_state=prior_var
        )
        # compute meas parameters
        wgt_meas, mean_meas, var_meas = interrogate(
            key=keys[1],
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=ode_time,
            mean_state_pred=mean_state_pred_z,
            var_state_pred=var_state_pred_z,
            **params
        )
        W_meas = ode_weight + wgt_meas
        # kalman forecast and update
        logp, mean_state_next_z, var_state_next_z = jax.vmap(forecast_update)(
                mean_state_pred=mean_state_pred_z,
                var_state_pred=var_state_pred_z,
                x_meas=x_meas,
                mean_meas=mean_meas,
                wgt_meas=W_meas,
                var_meas=var_meas
        )                       
        logdens_z += jnp.sum(logp)
        # carry over state
        carry = {
            "state_filt_joint": (mean_state_next_zy, var_state_next_zy),
            "state_filt_marg": (mean_state_next_z, var_state_next_z),
            "logdens_joint": logdens_zy,
            "logdens_marg": logdens_z,
            "i": i
        }
        return carry, None
    
    # compute log-density of p(Y_0 |X_0) if Y_0 is at time 0
    def _logy0():
        logdens_zy = jnp.sum(
            jax.vmap(lambda b:
                    multivariate_normal_logpdf(obs_data[0, b], mean=obs_weight[0, b].dot(ode_init[b]) + obs_mean[b], cov=obs_var[0, b])
            )(jnp.arange(n_block)))
        return logdens_zy, 1
    def _no_logy0():
        return 0.0, 0
    logdens_zy, i = jax.lax.cond(obs_ind[0] == 0, _logy0, _no_logy0)
    
    scan_init = {
        "state_filt_joint": (mean_state_init, var_state_init),
        "state_filt_marg": (mean_state_init, var_state_init),
        "logdens_joint": logdens_zy,
        "logdens_marg": 0.0,
        "i": i
    }

    if key is not None:
        keys = jax.random.split(key, num=(n_steps, 2))
    else:
        keys = jnp.zeros((n_steps, 2))

    filter_kwargs = {
        "t": jnp.arange(n_steps),
        "key": keys
    }
    out, _ = jax.lax.scan(scan, scan_init, filter_kwargs)
    return out["logdens_joint"] - out["logdens_marg"]


# --- ODE solver --------------------------------------------------------------


# use linearizations and observations
def _solve_filter(key, ode_fun, ode_weight, ode_init, 
                  t_min, t_max, n_steps,
                  interrogate,
                  prior_weight, prior_var,
                  obs_data, obs_times, obs_weight, obs_var,
                  kalman_funs, **params):
    r"""
    Forward pass of the DALTON algorithm with Gaussian observations. Same arguments as :func:`~dalton.dalton`.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = ode_weight.shape
    # Dimension of observation
    n_bobs = obs_weight.shape[2]

    # insert observations on solver time grid
    sim_times = jnp.linspace(t_min, t_max, n_steps + 1)
    obs_ind = jnp.searchsorted(sim_times, obs_times)

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    obs_mean = jnp.zeros((n_block, n_bobs))
    mean_state = jnp.zeros((n_block, n_bstate))
    mean_state_init = ode_init
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # compute p(X_{1:n} | Z_{1:n}, Y_{0:m})
    def scan_fun(carry, filter_kwargs):
        mean_state_filt, var_state_filt = carry["state_filt"]
        i = carry["i"]
        t = filter_kwargs["t"]
        key = filter_kwargs["key"]
        ode_time = t_min + (t_max-t_min)*(t+1)/n_steps
        
        # kalman predict
        mean_state_pred, var_state_pred = jax.vmap(kalman_funs.predict)(
                mean_state_past=mean_state_filt,
                var_state_past=var_state_filt,
                mean_state=mean_state,
                wgt_state=prior_weight,
                var_state=prior_var
        )
        # compute meas parameters
        wgt_meas, mean_meas, var_meas = interrogate(
            key=key,
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=ode_time,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            **params
        )
        W_meas = ode_weight + wgt_meas

        # both z and y are observed
        def zy_update():
            wgt_meas_obs = jnp.concatenate([W_meas, obs_weight[i]], axis=1)
            mean_meas_obs = jnp.concatenate([mean_meas, obs_mean], axis=1)
            var_meas_obs = jax.vmap(jsp.linalg.block_diag)(var_meas, obs_var[i])
            x_meas_obs = jnp.concatenate([x_meas, obs_data[i]], axis=1)
            mean_state_next, var_state_next = jax.vmap(kalman_funs.update)(
                    mean_state_pred=mean_state_pred,
                    var_state_pred=var_state_pred,
                    x_meas=x_meas_obs,
                    mean_meas=mean_meas_obs,
                    wgt_meas=wgt_meas_obs,
                    var_meas=var_meas_obs
            )                        
            return mean_state_next, var_state_next, i+1

        # only z is observed
        def z_update():
            mean_state_next, var_state_next = jax.vmap(kalman_funs.update)(
                    mean_state_pred=mean_state_pred,
                    var_state_pred=var_state_pred,
                    x_meas=x_meas,
                    mean_meas=mean_meas,
                    wgt_meas=W_meas,
                    var_meas=var_meas
            )                            
            return mean_state_next, var_state_next, i

        mean_state_next, var_state_next, i = jax.lax.cond(t+1 == obs_ind[i], zy_update, z_update)
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next),
            "i": i
        }
        stack = {
            "state_filt": (mean_state_next, var_state_next),
            "state_pred": (mean_state_pred, var_state_pred)
        }
        return carry, stack

    # check if observations start at 0
    i = jax.lax.cond(obs_ind[0] == 0, lambda: 1, lambda: 0)

    # scan initial value for computing p(X_{0:n} | Y_{0:m}, Z_{1:n})
    scan_init = {
        "state_filt": (mean_state_init, var_state_init),
        "i": i
    }
    
    if key is not None:
        keys = jax.random.split(key, num=n_steps)
    else:
        keys = jnp.zeros(n_steps)
    
    filter_kwargs = {
        "t": jnp.arange(n_steps),
        "key": keys
    }
    _, scan_out = jax.lax.scan(scan_fun, scan_init, filter_kwargs)
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


def solve_mv(key, ode_fun, ode_weight, ode_init, 
             t_min, t_max, n_steps,
             interrogate,
             prior_weight, prior_var,
             obs_data, obs_times, obs_weight, obs_var,
             kalman_funs=standard, **params):
    r"""
    DALTON algorithm to compute the mean and variance of :math:`p(X_{0:N} \mid Y_{0:M}, Z_{1:N})` assuming Gaussian observations.
    Same arguments as :func:`~dalton.dalton`.

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
        obs_data=obs_data, obs_times=obs_times,
        obs_weight=obs_weight, obs_var=obs_var,
        kalman_funs=kalman_funs, **params   
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        mean_state_curr, var_state_curr = jax.vmap(kalman_funs.smooth_mv)(
                mean_state_next=state_next["mean"],
                var_state_next=state_next["var"],
                wgt_state=prior_weight,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                var_State=prior_var
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
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs, reverse=True)

    # append initial values to front and back
    mean_state_smooth = jnp.concatenate(
        [ode_init[None], scan_out["mean"], scan_init["mean"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_block, n_bstate, n_bstate))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return mean_state_smooth, var_state_smooth


def solve_sim(key, ode_fun, ode_weight, ode_init, 
              t_min, t_max, n_steps,
              interrogate,
              prior_weight, prior_var,
              obs_data, obs_times, obs_weight, obs_var,
              kalman_funs=standard, **params):
    r"""
    DALTON algorithm to sample from :math:`p(X_{0:N} \mid Y_{0:M}, Z_{1:N})` assuming Gaussian observations.
    Same arguments as :func:`~dalton.dalton`.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate, _ = prior_weight.shape
    key, *subkeys = jax.random.split(key, num=n_steps+1)
    # forward pass
    filt_out = _solve_filter(
        key=key,
        ode_fun=ode_fun, ode_weight=ode_weight, ode_init=ode_init,
        t_min=t_min, t_max=t_max, n_steps=n_steps, 
        interrogate=interrogate,
        prior_weight=prior_weight, prior_var=prior_var,
        obs_data=obs_data, obs_times=obs_times,
        obs_weight=obs_weight, obs_var=obs_var,
        kalman_funs=kalman_funs, **params   
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

        mean_state_sim, var_state_sim = jax.vmap(kalman_funs.smooth_sim)(
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
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs, reverse=True)

    # append initial values to front and back
    x_state_smooth = jnp.concatenate(
        [ode_init[None], scan_out, scan_init[None]]
    )
    return x_state_smooth

# --- non-Gaussian loglikelihood -------------------------------------------------


def _solve_filter_nn(key, ode_fun, ode_weight, ode_init, 
                     t_min, t_max, n_steps,
                     interrogate,
                     prior_weight, prior_var,
                     obs_data, obs_times, obs_loglik_i,
                     kalman_funs, **params):
    r"""
    Forward pass of the DALTON algorithm using non-Gaussian observations. Same arguments as :func:`~dalton.daltonng`.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = ode_weight.shape
    # Dimension of observation
    # n_bobs = obs_data.shape[2]

    # insert observations on solver time grid
    sim_times = jnp.linspace(t_min, t_max, n_steps + 1)
    obs_ind = jnp.searchsorted(sim_times, obs_times)

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
    obs_mean = jnp.zeros((n_block, n_bstate))
    mean_state = jnp.zeros((n_block, n_bstate))
    mean_state_init = ode_init
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))

    # compute p(X_{1:n} | Z_{1:n}, \hat Y_{0:m})
    def scan_fun(carry, filter_kwargs):
        mean_state_filt, var_state_filt = carry["state_filt"]
        i = carry["i"]
        t = filter_kwargs["t"]
        key = filter_kwargs["key"]
        ode_time = t_min + (t_max-t_min)*(t+1)/n_steps
        
        # kalman predict
        mean_state_pred, var_state_pred = jax.vmap(kalman_funs.predict)(
            mean_state_past=mean_state_filt,
            var_state_past=var_state_filt,
            mean_state=mean_state,
            wgt_state=prior_weight,
            var_state=prior_var
        )
        # compute meas parameters
        wgt_meas, mean_meas, var_meas = interrogate(
            key=key,
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=ode_time,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            **params
        )
        W_meas = ode_weight + wgt_meas

        # both z and y are observed
        def zy_update():
            # transform to yhat
            obs_grad = jax.jacrev(obs_loglik_i, argnums=1)(obs_data[i], mean_state_pred, i, **params)
            obs_hes = jax.jacfwd(jax.jacrev(obs_loglik_i, argnums=1), argnums=1)(
                obs_data[i], mean_state_pred, i, **params
            )
            obs_var = jax.vmap(lambda b: -jnp.linalg.pinv(obs_hes[b, :, b]))(jnp.arange(n_block))
            obs_weight = jnp.where(obs_var != 0, 1, 0)
            obs_hat = jax.vmap(lambda b: 
                obs_weight[i].dot(mean_state_pred[b]) + obs_var[b].dot(obs_grad[b])
            )(jnp.arange(n_block))
            
            # Cmu = jax.vmap(lambda b: wgt_curr[b].dot(mean_state_pred[b]))(jnp.arange(n_block))
            # gpmu = jax.jacfwd(fun_obs)(Cmu, y_curr, theta, i)
            # gppmu = jax.jacfwd(jax.jacrev(fun_obs))(Cmu, y_curr, theta, i)
            # var_obs = jax.vmap(lambda b: -jnp.linalg.pinv(gppmu[b, :, b]))(jnp.arange(n_block))
            # y_new = jax.vmap(lambda b: Cmu[b] + var_obs[b].dot(gpmu[b]))(jnp.arange(n_block))
            # stack measure and observation variables
            wgt_meas_obs = jnp.concatenate([W_meas, obs_weight], axis=1)
            mean_meas_obs = jnp.concatenate([mean_meas, obs_mean], axis=1)
            var_meas_obs = jax.vmap(lambda b: jsp.linalg.block_diag(var_meas[b], obs_var[b]))(jnp.arange(n_block))
            x_meas_obs = jnp.concatenate([x_meas, obs_hat], axis=1)
            # kalman update
            mean_state_next, var_state_next = jax.vmap(kalman_funs.update)(
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                x_meas=x_meas_obs,
                mean_meas=mean_meas_obs,
                wgt_meas=wgt_meas_obs,
                var_meas=var_meas_obs
            )
            return mean_state_next, var_state_next, i+1
        
        # only z is observed
        def z_update():
            # kalman update
            mean_state_next, var_state_next = jax.vmap(kalman_funs.update)(
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                x_meas=x_meas,
                mean_meas=mean_meas,
                wgt_meas=W_meas,
                var_meas=var_meas
            )
            return mean_state_next, var_state_next, i

        mean_state_next, var_state_next, i = jax.lax.cond(t+1 == obs_ind[i], zy_update, z_update)
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next),
            "i": i
        }
        stack = {
            "state_filt": (mean_state_next, var_state_next),
            "state_pred": (mean_state_pred, var_state_pred)
        }
        return carry, stack

    # check if observations start at 0
    i = jax.lax.cond(obs_ind[0] == 0, lambda: 1, lambda: 0)

    # scan initial value for computing p(X_{0:n} | \hat Y_{0:m}, Z_{1:n})
    scan_init = {
        "state_filt": (mean_state_init, var_state_init),
        "i": i
    }
    
    if key is not None:
        keys = jax.random.split(key, num=n_steps)
    else:
        keys = jnp.zeros(n_steps)
    
    filter_kwargs = {
        "t": jnp.arange(n_steps),
        "key": keys
    }
    _, scan_out = jax.lax.scan(scan_fun, scan_init, filter_kwargs)
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


def _logx_yhat(mean_state_filt, var_state_filt,
               mean_state_pred, var_state_pred,
               prior_weight, prior_var, kalman_funs):
    r"""
    Compute the loglikelihood of :math:`p(X_{0:N} \mid \hat Y_{0:M}, Z_{1:N})`.
    
    Args:
        mean_state_pred (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        mean_state_filt (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n}`.
        var_state_filt (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
        prior_weight (ndarray(n_block, n_bstate, n_bstate)): Weight matrix defining the solution prior; :math:`Q`.
        prior_var (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        kalman_funs (object): An object that contains the Kalman filtering functions: predict, update and smooth.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`p(X_{0:N} \mid \hat Y_{0:M}, Z_{1:N})`.
        - **logx_yhat** (float): Loglikelihood of :math:`p(X_{0:N} \mid \hat Y_{0:M}, Z_{1:N})`.
    """
    # dimensions
    n_tot, n_block, _ = mean_state_filt.shape
    n_steps = n_tot - 1

    # backward pass
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        logx_yhat = state_next["logx_yhat"]
        mean_state_curr, var_state_curr = jax.vmap(kalman_funs.smooth_mv)(
            mean_state_next=state_next["mean"],
            var_state_next=state_next["var"],
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=prior_weight,
            var_state=prior_var
        )
        mean_state_sim, var_state_sim = jax.vmap(kalman_funs.smooth_sim)(
            x_state_next=state_next["mean"],
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=prior_weight
        )
        logx_yhat += jnp.sum(
            jax.vmap(multivariate_normal_logpdf)(mean_state_curr, mean=mean_state_sim, cov=var_state_sim)
        )
        carry = {
            "mean": mean_state_curr,
            "var": var_state_curr,
            "logx_yhat":  logx_yhat
        }
        return carry, carry
    # compute log(mu_{N|N}) at the last filtering step
    logx_yhatN = jnp.sum(
        jax.vmap(multivariate_normal_logpdf)(mean_state_filt[n_steps], mean=mean_state_filt[n_steps], cov=var_state_filt[n_steps])
    )
    # initialize
    scan_init = {
        "mean": mean_state_filt[n_steps],
        "var": var_state_filt[n_steps],
        "logx_yhat": logx_yhatN
    }
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    last_scan, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,reverse=True)

    # append initial values to front and terminal value to the back
    scan_out["mean"] = jnp.concatenate(
        [mean_state_filt[0][None], scan_out["mean"], scan_init["mean"][None]]
    )
    return scan_out["mean"], last_scan["logx_yhat"]


def _logx_z(uncond_mean, 
            mean_state_filt, var_state_filt,
            mean_state_pred, var_state_pred,
            prior_weight, kalman_funs):
    r"""
    Compute the loglikelihood of :math:`p(X_{0:N} \mid Z_{1:N})`.
    
    Args:
        uncond_mean (ndarray(n_steps+1, n_block, n_bstate)): Unconditional mean computed from :math:`p(X_{0:N} \mid \hat Y_{0:M}, Z_{1:N})`.
        mean_state_pred (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        mean_state_filt (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n}`.
        var_state_filt (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.
        prior_weight (ndarray(n_block, n_bstate, n_bstate)): Weight matrix defining the solution prior; :math:`Q`.
        kalman_funs (object): An object that contains the Kalman filtering functions: predict, update and smooth.
    
    Return:
        (float): Loglikelihood of :math:`p(X_{0:N} \mid Z_{1:N})`.

    """
    # dimensions
    n_tot, n_block, _ = mean_state_filt.shape
    n_steps = n_tot - 1

    # backward pass
    def scan_fun(logx_z, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        uncond_next = smooth_kwargs['uncond_next']
        uncond_curr = smooth_kwargs['uncond_curr']
        mean_state_sim, var_state_sim = jax.vmap(kalman_funs.smooth_sim)(
            x_state_next=uncond_next,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=prior_weight
        )
        logx_z += jnp.sum(
            jax.vmap(multivariate_normal_logpdf)(uncond_curr, mean=mean_state_sim, cov=var_state_sim)
        )
        return logx_z, logx_z
    # compute log(mu_{N|N}) at the last filtering step
    logx_zN = jnp.sum(
        jax.vmap(multivariate_normal_logpdf)(uncond_mean[n_steps], mean=mean_state_filt[n_steps], cov=var_state_filt[n_steps])
    )
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
    scan_out, _ = jax.lax.scan(scan_fun, logx_zN, scan_kwargs, reverse=True)
    
    return scan_out

def daltonng(key, ode_fun, ode_weight, ode_init, 
             t_min, t_max, n_steps,
             interrogate,
             prior_weight, prior_var,
             obs_data, obs_times, obs_loglik_i,
             kalman_funs=standard, **params):
    r"""
    Compute marginal loglikelihood of DALTON algorithm for non-Gaussian observations; :math:`p(\hat Y_{0:M} \mid Z_{1:N})`.

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
        obs_data (ndarray(n_obs, n_blocks, n_bobs)): Observed data; :math:`Y_{0:M}`.
        obs_times (ndarray(n_obs)): Observation time; :math:`0, \ldots, M`.
        obs_loglik_i (Callable): Loglikelihood function for each observation.
        kalman_funs (object): An object that contains the Kalman filtering functions: predict, update and smooth.
        params (kwargs): Optional model parameters.

    
    Returns:
        (float): Loglikelihood of :math:`p(\hat Y_{0:M} \mid Z_{1:N})`.

    """
    n_obs = obs_data.shape[0]

    # forward pass
    filt_out = _solve_filter_nn(
        key=key,
        ode_fun=ode_fun, ode_weight=ode_weight, ode_init=ode_init,
        t_min=t_min, t_max=t_max, n_steps=n_steps, 
        interrogate=interrogate,
        prior_weight=prior_weight, prior_var=prior_var,
        obs_data=obs_data, obs_times=obs_times,
        obs_loglik_i=obs_loglik_i,
        kalman_funs=kalman_funs, **params   
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # logp(X_{0:N} | \hat Y_{0:M}, Z_{1:N})
    mean_state_smooth, logx_yhat = _logx_yhat(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        prior_weight=prior_weight,
        prior_var=prior_var,
        kalman_funs=kalman_funs
    )

    # logp(Y_{0:M} | X_{0:M})
    sim_times = jnp.linspace(t_min, t_max, n_steps+1)
    obs_ind = jnp.searchsorted(sim_times, obs_times)
    def vmap_fun(i):
        # Cx = jax.vmap(lambda b: mean_state_smooth[obs_ind[i]][b])(jnp.arange(n_block))
        # return fun_obs(Cx, y_obs[i], theta, i)
        return obs_loglik_i(obs_data[i], mean_state_smooth[obs_ind[i]], i, **params)
    logy_x = jnp.sum(jax.vmap(vmap_fun)(jnp.arange(n_obs)))

    # logp(X_{0:N} | Z_{1:N})
    # first do forward pass without obs
    filt_out = _solve_filter_ode(
        key=key,
        ode_fun=ode_fun, ode_weight=ode_weight, ode_init=ode_init,
        t_min=t_min, t_max=t_max, n_steps=n_steps, 
        interrogate=interrogate,
        prior_weight=prior_weight, prior_var=prior_var,
        kalman_funs=kalman_funs, **params   
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    logx_z = _logx_z(
        uncond_mean=mean_state_smooth,
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        prior_weight=prior_weight,
        kalman_funs=kalman_funs
    )

    return logy_x + logx_z - logx_yhat


# --- non-Gaussian ODE solver -------------------------------------------------


def solve_mv_nn(key, ode_fun, ode_weight, ode_init, 
                t_min, t_max, n_steps,
                interrogate,
                prior_weight, prior_var,
                obs_data, obs_times, obs_loglik_i,
                kalman_funs=standard, **params):
    r"""
    DALTON algorithm to compute the mean and variance of :math:`p(X_{0:N} \mid \hat Y_{0:M}, Z_{1:N})` assuming non-Gaussian observations. 
    Same arguments as :func:`~dalton.daltonng`.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate, _ = prior_weight.shape
    # forward pass
    filt_out = _solve_filter_nn(
        key=key,
        ode_fun=ode_fun, ode_weight=ode_weight, ode_init=ode_init,
        t_min=t_min, t_max=t_max, n_steps=n_steps, 
        interrogate=interrogate,
        prior_weight=prior_weight, prior_var=prior_var,
        obs_data=obs_data, obs_times=obs_times,
        obs_loglik_i=obs_loglik_i,
        kalman_funs=kalman_funs, **params   
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]
    # backward pass
    def scan_fun(state_next, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        mean_state_curr, var_state_curr = jax.vmap(kalman_funs.smooth_mv)(
            mean_state_next=state_next["mean"],
            var_state_next=state_next["var"],
            wgt_state=prior_weight,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            var_state=prior_var
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
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs, reverse=True)

    # append initial values to front and back
    mean_state_smooth = jnp.concatenate(
        [ode_init[None], scan_out["mean"], scan_init["mean"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_block, n_bstate, n_bstate))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return mean_state_smooth, var_state_smooth
