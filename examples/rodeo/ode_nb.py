r"""
Stochastic solver for ODE initial value problems.

The ODE-IVP to be solved is defined as

.. math:: W X_t = F(X_t, t, \theta)

on the time interval :math:`t \in [a, b]` with initial condition :math:`X_a = x_0`.  

The stochastic solver proceeds via Kalman filtering and smoothing of "interrogations" of the ODE model as described in Chkrebtii et al 2016, Schober et al 2019.  In the context of the underlying Kalman filterer/smoother, the Gaussian state-space model is

:: math::

X_n = Q X_{n-1} + R^{1/2} \epsilon_n

y_n = W X_n + \Sigma_n \eta_n,

where :math:`\epsilon_n` and :math:`\eta_n` are iid standard normals the size of :math:`X_t` and :math:`W X_t`, respectively, and where :math:`(y_n, \Sigma_n)` are generated sequentially using the ODE function :math:`F(X_t, t, \theta)` as explained in the references above.  Thus, much of the notation here is identical to that in the `kalmantv` module.
  
"""

# import numpy as np
import jax
import jax.numpy as jnp
from rodeo.kalmantv import *


def interrogate_rodeo(key, fun, W, t, theta,
                      mean_state_pred, var_state_pred):
    r"""
    Rodeo interrogation method.

    Args:
        key (PRNGKey): Jax PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = f(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_meas, n_state)): Transition matrix defining the measure prior.
        t (float): Time point.
        theta (ndarray(n_theta)): ODE parameter.
        mean_state_pred (ndarray(n_state)): Mean estimate for state at time t given observations from times [a...t-1]; denoted by :math:`\mu_{t|t-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time t given observations from times [a...t-1]; denoted by :math:`\Sigma_{t|t-1}`.

    Returns:
        (tuple):
        - **trans_meas** (ndarray(n_meas, n_state)): Interrogation transition matrix.
        - **mean_meas** (ndarray(n_meas)): Interrogation offset.
        - **var_meas** (ndarray(n_meas, n_meas)): Interrogation variance.

    """
    var_meas = jnp.atleast_2d(
        jnp.linalg.multi_dot([W, var_state_pred, W.T])
    )
    x_state = mean_state_pred
    mean_meas = -fun(x_state, t, theta)
    return jnp.zeros(W.shape), mean_meas, var_meas

def interrogate_chkrebtii(key, fun, W, t, theta,
                          mean_state_pred, var_state_pred):
    r"""
    Interrogate method of Chkrebtii et al (2016); DOI: 10.1214/16-BA1017.

    Same arguments and returns as :func:`~ode_nb.interrogate_rodeo`.

    """
    #key, subkey = jax.random.split(key)
    #z_state = jax.random.normal(key, (n_state, ))
    var_meas = jnp.atleast_2d(
        jnp.linalg.multi_dot([W, var_state_pred, W.T])
    )
    x_state = jax.random.multivariate_normal(key, mean_state_pred, var_state_pred)
    mean_meas = -fun(x_state, t, theta)
    return jnp.zeros(W.shape), mean_meas, var_meas

def interrogate_schober(key, fun, t, theta,
                        W, mean_state_pred, var_state_pred):
    r"""
    Interrogate method of Schober et al (2019); DOI: https://doi.org/10.1007/s11222-017-9798-7.

    Same arguments and returns as :func:`~ode_nb.interrogate_rodeo`.

    """
    n_meas = W.shape[0]
    var_meas = jnp.zeros((n_meas, n_meas))
    x_state = mean_state_pred
    mean_meas = -fun(x_state, t, theta)
    return jnp.zeros(W.shape), mean_meas, var_meas

def interrogate_kramer(key, fun, W, t, theta,
                        mean_state_pred, var_state_pred):
    r"""
    First order interrogate method of Tronarp et al (2019); DOI: https://doi.org/10.1007/s11222-019-09900-1.
    Same arguments and returns as :func:`~ode.interrogate_rodeo`.

    """
    n_meas, n_state = W.shape
    p = int(n_state/n_meas)
    fun_meas = -fun(mean_state_pred, t, theta)
    jac = jax.jacfwd(fun)(mean_state_pred, t, theta)
    trans_meas = -jac
    mean_meas = fun_meas + jac.dot(mean_state_pred)
    # var_meas = jax.vmap(lambda wm, vsp:
    #                     jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
    #     trans_meas, var_state_pred
    # )
    var_meas = jnp.zeros((n_meas, n_meas))
    return trans_meas, mean_meas, var_meas



def _solve_filter(key, fun, W, x0, theta,
                  tmin, tmax, n_steps,
                  trans_state, var_state,
                  interrogate=interrogate_rodeo):
    r"""
    Forward pass of the ODE solver.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_meas, n_state = W.shape

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_meas))
    mean_state = jnp.zeros((n_state))
    mean_state_init = x0
    var_state_init = jnp.zeros((n_state, n_state))
    
    # lax.scan setup
    # scan function
    def scan_fun(carry, t):
        mean_state_filt, var_state_filt = carry["state_filt"]
        key, subkey = jax.random.split(carry["key"])
        # kalman predict
        mean_state_pred, var_state_pred = predict(
            mean_state_past=mean_state_filt,
            var_state_past=var_state_filt,
            mean_state=mean_state,
            trans_state=trans_state,
            var_state=var_state
        )
        # model interrogation
        trans_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            t=tmin + (tmax-tmin)*(t+1)/n_steps,
            theta=theta,
            W=W,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        W_meas = W + trans_meas
        # kalman update
        mean_state_next, var_state_next = update(
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            x_meas=x_meas,
            mean_meas=mean_meas,
            trans_meas=W_meas,
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


def solve_sim(key, fun, W, x0, theta,
              tmin, tmax, n_steps,
              trans_state, var_state,
              interrogate=interrogate_rodeo):
    r"""
    Random draw from the stochastic ODE solver.

    Args:
        key: PRNG key.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        x0 (float): Initial value of the state variable :math:`x_t` at time :math:`t = 0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        W (ndarray(n_state)): Transition matrix defining the measure prior; :math:`W`.
        interrogate: Function defining the interrogation method.

    Returns:
        **x_state_smooth** (ndarray(n_steps, n_state)): Sample solution at time t given observations from times [0...N] for :math:`t = 0,1/N,\ldots,1`.

    """
    n_state = trans_state.shape[0]
    key, *subkeys = jax.random.split(key, num=n_steps+1)
    subkeys = jnp.array(subkeys)
    # z_state = jax.random.normal(subkey, (n_steps, n_state))

    # forward pass
    filt_out = _solve_filter(
        key=key,
        fun=fun, theta=theta, x0=x0,
        tmin=tmin, tmax=tmax, n_steps=n_steps,
        W=W, trans_state=trans_state,
        var_state=var_state,
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

        mean_state_sim, var_state_sim = smooth_sim(
            x_state_next=x_state_next,
            trans_state=trans_state,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        x_state_curr = jax.random.multivariate_normal(key, mean_state_sim, var_state_sim)
        return x_state_curr, x_state_curr
    # initialize
    #scan_init = _state_sim(mean_state_filt[n_steps],
    #                       var_state_filt[n_steps],
    #                       z_state[n_steps-1])
    scan_init = jax.random.multivariate_normal(subkeys[n_steps-1], 
                                               mean_state_filt[n_steps], 
                                               var_state_filt[n_steps])
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
    # kwargs_init = {
    #     'mean_state_filt': mean_state_filt[n_steps],
    #     'var_state_filt': var_state_filt[n_steps],
    #     'mean_state_pred': mean_state_pred[n_steps+1],
    #     'var_state_pred': var_state_pred[n_steps+1],
    #     'z_state': z_state[n_steps-1]}
    # scan_fun(scan_init, kwargs_init)
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    x_state_smooth = jnp.concatenate(
        [x0[None], scan_out, scan_init[None]]
    )
    return x_state_smooth


def solve_mv(key, fun, W, x0, theta,
             tmin, tmax, n_steps,
             trans_state, var_state,
             interrogate=interrogate_rodeo):
    r"""
    Mean and variance of the stochastic ODE solver.

    Args:
        key: PRNG key.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        x0 (float): Initial value of the state variable :math:`x_t` at time :math:`t = 0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        W (ndarray(n_state)): Transition matrix defining the measure prior; :math:`W`.
        interrogate: Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps, n_state)): Posterior mean of the solution process :math:`X_t` at times
          :math:`t = 0,1/N,\ldots,1`.
        - **var_state_smooth** (ndarray(n_steps, n_state, n_state)): Posterior variance of the solution process at
          times :math:`t = 0,1/N,\ldots,1`.

    """
    n_state = trans_state.shape[0]
    # forward pass
    # key, subkey = jax.random.split(key)
    filt_out = _solve_filter(
        key=key,
        fun=fun, theta=theta, x0=x0,
        tmin=tmin, tmax=tmax, n_steps=n_steps,
        W=W, trans_state=trans_state,
        var_state=var_state,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    # lax.scan setup
    def scan_fun(state_next, smooth_kwargs):
        mean_state_curr, var_state_curr = smooth_mv(
            mean_state_next=state_next["mu"],
            var_state_next=state_next["var"],
            trans_state=trans_state,
            **smooth_kwargs
            # mean_state_filt=mean_state_filt[t],
            # var_state_filt=var_state_filt[t],
            # mean_state_pred=var_state_pred[t+1],
            # var_state_pred=var_state_pred[t+1],
            # trans_state=trans_state
        )
        state_curr = {
            "mu": mean_state_curr,
            "var": var_state_curr
        }
        return state_curr, state_curr
    # initialize
    scan_init = {
        "mu": mean_state_filt[n_steps],
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
        [x0[None], scan_out["mu"], scan_init["mu"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_state, n_state))[None], scan_out["var"],
         scan_init["var"][None]]
    )
    return mean_state_smooth, var_state_smooth


def solve(key, fun, W, x0, theta,
          tmin, tmax, n_steps,
          trans_state, var_state,
          interrogate=interrogate_rodeo):
    r"""
    Both random draw and mean/variance of the stochastic ODE solver. 

    Args:
        key: PRNG key.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        x0 (float): Initial value of the state variable :math:`x_t` at time :math:`t = 0`.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`T`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        W (ndarray(n_state)): Transition matrix defining the measure prior; :math:`W`.
        interrogate: Function defining the interrogation method.

    Returns:
        **x_state_smooth** (ndarray(n_steps, n_state)): Sample solution at time t given observations from times [0...N] for
          :math:`t = 0,1/N,\ldots,1`.

    """
    n_state = trans_state.shape[0]
    key, *subkeys = jax.random.split(key, num=n_steps+1)
    subkeys = jnp.array(subkeys)

    # forward pass
    #key, subkey = jax.random.split(key)
    filt_out = _solve_filter(
        key=key,
        fun=fun, theta=theta, x0=x0,
        tmin=tmin, tmax=tmax, n_steps=n_steps,
        W=W, trans_state=trans_state,
        var_state=var_state,
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
        key = smooth_kwargs['key']
        mean_state_sim, var_state_sim, mean_state_curr, var_state_curr = smooth(
            x_state_next=state_next["x"],
            mean_state_next=state_next["mu"],
            var_state_next=state_next["var"],
            trans_state=trans_state,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred
        )
        x_state_curr = jax.random.multivariate_normal(key, mean_state_sim, var_state_sim)
        state_curr = {
            "x": x_state_curr,
            "mu": mean_state_curr,
            "var": var_state_curr
        }
        return state_curr, state_curr
    # initialize
    scan_init = {
        "x": jax.random.multivariate_normal(subkeys[n_steps-1], 
                                            mean_state_filt[n_steps],
                                            var_state_filt[n_steps]),
        "mu": mean_state_filt[n_steps],
        "var": var_state_filt[n_steps]
    }
    # scan arguments
    # Slice these arrays so they are aligned.
    # More precisely, for time step t, want filt[t], pred[t+1], z_state[t-1]
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:n_steps],
        'var_state_filt': var_state_filt[1:n_steps],
        'mean_state_pred': mean_state_pred[2:n_steps+1],
        'var_state_pred': var_state_pred[2:n_steps+1],
        'key': subkeys[:n_steps-1]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    _, scan_out = jax.lax.scan(scan_fun, scan_init, scan_kwargs,
                               reverse=True)

    # append initial values to front and back
    x_state_smooth = jnp.concatenate(
        [x0[None], scan_out["x"], scan_init["x"][None]]
    )
    mean_state_smooth = jnp.concatenate(
        [x0[None], scan_out["mu"], scan_init["mu"][None]]
    )
    var_state_smooth = jnp.concatenate(
        [jnp.zeros((n_state, n_state))[None], scan_out["var"],
         scan_init["var"][None]]
    )

    return x_state_smooth, mean_state_smooth, var_state_smooth
