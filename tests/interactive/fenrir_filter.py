r"""
The forward pass model is

.. math::

x_0 = v

x_n = Q x_{n-1} + R^{1/2} \epsilon_n

z_n = W x_n - f(x_n, t_n) + V_n^{1/2} \eta_n.

The reverse pass model is

.. math::

x_N \sim N(b_N^S, V_N^S)

x_n = A_n^S x_{n+1} + b_n^S + (V_n^S)^{1/2} \eta_n

y_n = C x_n + Omega^{1/2} \epsilon_n.

"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.kalmantv import *
from rodeo.ode import interrogate_tronarp, interrogate_rodeo

# def zero_update(fun, t, W, theta,
#                 mean_state_pred, var_state_pred):
    
#     n_meas = W.shape[0]
#     mean_meas = -fun(mean_state_pred, t, theta)
#     trans_meas = W
#     # var_meas = jnp.zeros((n_meas, n_meas))
#     var_meas = jnp.linalg.multi_dot([W, var_state_pred, W.T])
#     return mean_meas, trans_meas, var_meas

# use interrogations first then observations
def forward(key, fun, W, x0, theta,
            tmin, tmax, n_steps,
            trans_state, mean_state, var_state, interrogate):
    r"""
    Forward pass of the Fenrir algorithm.

    Args:
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        W (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        varzero (bool): Indicator for the variance of the measurement variable; :math:`z`.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape

    # arguments for kalman_filter and kalman_smooth
    x_meas = jnp.zeros((n_block, n_bmeas))
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

def backward(mean_state_filt, var_state_filt, 
             mean_state_pred, var_state_pred,
             trans_state):
    r"""
    Compute the backward markov chain parameters.

    Args:
        mean_state_pred (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        var_state_pred (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        mean_state_filt (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        var_state_filt (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        
    Returns:
        (tuple):
        - **trans_state_cond** (ndarray(n_steps, n_state, n_state)): Transition of smooth conditional at time t given observations from times [0...T]; :math:`A_{n|N}`.
        - **mean_state_cond** (ndarray(n_steps+1, n_state)): Offset of smooth conditional at time t given observations from times [0...T]; :math:`b_{n|N}`.
        - *var_state_cond** (ndarray(n_steps+1, n_state, n_state)): Variance of smooth conditional at time t given observations from times [0...T]; :math:`V_{n|N}`.

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

def reverse(trans_state, mean_state, var_state,
            trans_obs, mean_obs, var_obs, y_obs):
    
    r"""
    Backward pass of Fenrir algorithm. Uses observations.

    Args:
        trans_state (ndarray(n_steps, n_state, n_state)): Transition matrix defining the solution prior; :math:`A^S`.
        mean_state (ndarray(n_steps+1, n_state)): Transition_offsets defining the solution prior; :math:`b^S`.
        var_state (ndarray(n_steps+1, n_state, n_state)): Variance matrix defining the solution prior; :math:`V^S`.
        mean_obs (ndarray(n_obs)): Transition offsets defining the noisy observations.
        trans_obs (ndarray(n_obs, n_state)): Transition matrix defining the noisy observations; :math:`C`.
        var_obs (ndarray(n_obs, n_obs)): Variance matrix defining the noisy observations; :math:`Omega`.
        y_obs (ndarray(n_steps, n_obs)): Observed data; :math:`y_n`.

    Returns:
        logdens : The logdensity of p(Y_{0:N}).

    """
    # Add point to beginning of state variable for simpler loop
    n_tot, n_block, n_bstate = mean_state.shape
    trans_state_end = jnp.zeros(trans_state.shape[1:])
    trans_state = jnp.concatenate([trans_state, trans_state_end[None]])

    # reverse pass
    def scan_fun(carry, rev_args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        trans_state = rev_args['trans_state']
        mean_state = rev_args['mean_state']
        var_state = rev_args['var_state']
        y_obs = rev_args['y_obs']

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
            return mean_state_filt, var_state_filt

        # y_obs is not None
        def _obs():
            # kalman update
            mean_state_filt, var_state_filt = jax.vmap(lambda b:
                update(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    W = trans_obs[b],
                    x_meas=y_obs[b],
                    mean_meas = mean_obs[b],
                    trans_meas = trans_obs[b],
                    var_meas=var_obs[b]
                )
            )(jnp.arange(n_block))
            return mean_state_filt, var_state_filt

        mean_state_filt, var_state_filt = jax.lax.cond(jnp.isnan(y_obs).any(), _no_obs, _obs)

        carry = {
            "state_filt": (mean_state_filt, var_state_filt)
        }
        stack = {
            "state_pred": (mean_state_pred, var_state_pred),
            "state_filt": (mean_state_filt, var_state_filt)
        }
        return carry, stack

    # start at N+1 assuming 0 mean and variance
    # scan_init = {
    #     "state_filt" : (jnp.zeros((n_state,)), jnp.zeros((n_state, n_state))),
    #     "logdens" : 0.0
    # }
    scan_init = {
        "state_filt" : (jnp.zeros((n_block, n_bstate,)), jnp.zeros((n_block, n_bstate, n_bstate)))
    }
    rev_args = {
        "trans_state" : trans_state,
        "mean_state" : mean_state,
        "var_state" : var_state,
        "y_obs": y_obs
    }

    scan_out, scan_out2 = jax.lax.scan(scan_fun, scan_init, rev_args, reverse=True)
    # return scan_out["logdens"]
    return scan_out2

def smooth(trans_state, state_par):
    r"""
    Kalman smooth for the final pass in Fenrir.
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

def fenrir_filter(key, fun, W, x0, theta, tmin, tmax, n_res,
                  trans_state, mean_state, var_state,
                  trans_obs, mean_obs, var_obs, y_obs, interrogate=interrogate_rodeo):
    
    r"""
    Fenrir algorithm.

    Args:
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        W (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        mean_obs (ndarray(n_obs)): Transition offsets defining the noisy observations.
        trans_obs (ndarray(n_obs, n_state)): Transition matrix defining the noisy observations; :math:`C`.
        var_obs (ndarray(n_obs, n_obs)): Variance matrix defining the noisy observations; :math:`Omega`.
        y_obs (ndarray(n_steps, n_obs)): Observed data; :math:`y_n`.


    Returns:
        logdens : The logdensity of p(y_{0:N}).

    """
    n_obs, n_block, n_bmeas = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_block, n_bmeas))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)
    # forward pass
    filt_out = forward(
        key=key, fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, n_steps=n_steps,
        trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        interrogate=interrogate
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    trans_state_cond, mean_state_cond, var_state_cond = backward(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        trans_state=trans_state
    )

    # reverse pass
    # logdens = reverse(
    #     trans_state=trans_state_cond, mean_state=mean_state_cond, 
    #     var_state=var_state_cond, wgt_obs=wgt_obs,
    #     mu_obs=mu_obs, var_obs=var_obs, y_obs=y_obs
    # )
    state_par = reverse(
        trans_state=trans_state_cond, mean_state=mean_state_cond, 
        var_state=var_state_cond, trans_obs=trans_obs,
        mean_obs=mean_obs, var_obs=var_obs, y_obs=y_obs
    )
    mean_state_smooth, var_state_smooth = smooth(trans_state_cond, state_par)
    return mean_state_smooth, var_state_smooth
    # return state_par

# use data first then interrogations
def obs_passx(x0, trans_state, mean_state, var_state,
              trans_obs, mean_obs, var_obs, y_obs):
    r"""
    Forward pass of the Fenrir algorithm.

    Args:
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        W (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        varzero (bool): Indicator for the variance of the measurement variable; :math:`z`.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_state = len(mean_state)

    # arguments for forward
    mean_state_init = x0
    var_state_init = jnp.zeros((n_state, n_state))


    # forward pass using data
    def scan_fun(carry, rev_args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        y_obs = rev_args['y_obs']

        # kalman predict
        mean_state_pred, var_state_pred = predict(
            mean_state_past=mean_state_filt,
            var_state_past=var_state_filt,
            mean_state=mean_state,
            trans_state=trans_state,
            var_state=var_state
        )
        # y_obs is None
        def _no_obs():
            mean_state_filt = mean_state_pred
            var_state_filt = var_state_pred
            return mean_state_filt, var_state_filt

        # y_obs is not None
        def _obs():
            # kalman update
            mean_state_filt, var_state_filt = update(
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                x_meas=y_obs,
                mean_meas = mean_obs,
                trans_meas = trans_obs,
                var_meas=var_obs
            )
            return mean_state_filt, var_state_filt

        mean_state_filt, var_state_filt = jax.lax.cond(jnp.isnan(y_obs).any(), _no_obs, _obs)

        # output
        carry = {
            "state_filt": (mean_state_filt, var_state_filt)
        }
        stack = {
            "state_pred": (mean_state_pred, var_state_pred),
            "state_filt": (mean_state_filt, var_state_filt)
        }
        return carry, stack


    scan_init = {
        "state_filt" : (mean_state_init, var_state_init)
    }
    rev_args = {
        "y_obs": y_obs[1:]
    }

    _, scan_out = jax.lax.scan(scan_fun, scan_init, rev_args)

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

def meas_passx(fun, W, x0, theta,
               tmin, tmax, n_steps,
               trans_state, mean_state, var_state):
    r"""
    Backward pass using interrogations.

    Args:
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        W (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        varzero (bool): Indicator for the variance of the measurement variable; :math:`z`.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_state)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_meas, n_state = W.shape

    # arguments for reverse pass using interrogations
    z_meas = jnp.zeros(n_meas)
    n_state = trans_state.shape[1]
    trans_state_end = jnp.zeros(trans_state.shape[1:])
    trans_state = jnp.concatenate([trans_state, trans_state_end[None]])

    # reverse pass
    def scan_fun(carry, rev_args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        trans_state = rev_args['trans_state']
        mean_state = rev_args['mean_state']
        var_state = rev_args['var_state']
        t = rev_args['t']

        # kalman predict
        mean_state_pred, var_state_pred = predict(
            mean_state_past=mean_state_filt,
            var_state_past=var_state_filt,
            mean_state=mean_state,
            trans_state=trans_state,
            var_state=var_state
        )
        # compute meas parameters
        mean_meas, trans_meas, var_meas = zero_update(
            fun = fun, 
            t = t,
            theta = theta,
            W = W, 
            mean_state_pred = mean_state_pred, 
            var_state_pred = var_state_pred
        )
        # kalman update
        mean_state_next, var_state_next = update(
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            x_meas=z_meas,
            mean_meas=mean_meas,
            trans_meas=trans_meas,
            var_meas=var_meas
        )
        # output
        carry = {
            "state_filt": (mean_state_next, var_state_next)
        }
        stack = {
            "state_filt": (mean_state_next, var_state_next),
            "state_pred": (mean_state_pred, var_state_pred)
        }
        return carry, stack

    # start at N+1 assuming 0 mean and variance
    scan_init = {
        "state_filt" : (jnp.zeros((n_state,)), jnp.zeros((n_state, n_state)))
    }
    rev_args = {
        "trans_state" : trans_state[1:],
        "mean_state" : mean_state[1:],
        "var_state" : var_state[1:],
        "t": jnp.linspace(tmin, tmax, n_steps+1)[1:]
    }
    # scan itself
    _, scan_out = jax.lax.scan(scan_fun, scan_init, rev_args, reverse=True)
    # append initial values to front
    scan_out["state_filt"] = (
        jnp.concatenate([x0[None], scan_out["state_filt"][0]]),
        jnp.concatenate([jnp.zeros((n_state, n_state))[None], scan_out["state_filt"][1]])
    )
    scan_out["state_pred"] = (
        jnp.concatenate([x0[None], scan_out["state_pred"][0]]),
        jnp.concatenate([jnp.zeros((n_state, n_state))[None], scan_out["state_pred"][1]])
    )
    return scan_out

def rfenrir_filter(fun, W, x0, theta, tmin, tmax, n_res,
                  trans_state, mean_state, var_state,
                  trans_obs, mean_obs, var_obs, y_obs):
    
    r"""
    Fenrir algorithm like but uses data first instead of interrogations.

    Args:
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        W (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        mean_obs (ndarray(n_obs)): Transition offsets defining the noisy observations.
        trans_obs (ndarray(n_obs, n_state)): Transition matrix defining the noisy observations; :math:`C`.
        var_obs (ndarray(n_obs, n_obs)): Variance matrix defining the noisy observations; :math:`Omega`.
        y_obs (ndarray(n_steps, n_obs)): Observed data; :math:`y_n`.


    Returns:
        logdens : The logdensity of p(y_{0:N}).

    """
    n_obs, n_dim_obs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_dim_obs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)

    # forward pass using data
    filt_out = obs_passx(
        x0=x0, 
        trans_state=trans_state, mean_state=mean_state, var_state=var_state, 
        trans_obs=trans_obs, mean_obs=mean_obs, var_obs=var_obs, y_obs=y_obs
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]
    # backward pass
    trans_state_cond, mean_state_cond, var_state_cond = backward(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        trans_state=trans_state
    )
    # reverse pass using interrogations
    state_par = meas_passx(
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, n_steps=n_steps,
        trans_state=trans_state_cond,
        mean_state=mean_state_cond, var_state=var_state_cond
    )
    mean_state_smooth, var_state_smooth = smooth(trans_state_cond, state_par)
    return filt_out, state_par, mean_state_smooth, var_state_smooth
    # return state_par



# for non-gaussian observaions

def reverseng(trans_state, mean_state, var_state,
              wgt_obs, y_obs, gamma, theta,
              obs_fun, x_fun):
    
    r"""
    Reverse pass for non-gaussian.

    Args:
        trans_state (ndarray(n_steps-1, n_state, n_state)): Transition matrix defining the solution prior; :math:`A^S`.
        mean_state (ndarray(n_steps, n_state)): Transition_offsets defining the solution prior; :math:`b^S`.
        var_state (ndarray(n_steps, n_state, n_state)): Variance matrix defining the solution prior; :math:`V^S`.
        mu_obs (ndarray(n_obs)): Transition offsets defining the noisy observations.
        wgt_obs (ndarray(n_obs, n_state)): Transition matrix defining the noisy observations; :math:`C`.
        var_obs (ndarray(n_obs, n_obs)): Variance matrix defining the noisy observations; :math:`Omega`.
        y_obs (ndarray(n_steps, n_obs)): Observed data; :math:`y_n`.

    Returns:
        logdens : The logdensity of p(X_{0:N}, Y_{0:N}).

    """
    # # Add point to end of state variable for simpler loop
    n_state = trans_state.shape[1]
    trans_state_end = jnp.zeros(trans_state.shape[1:])
    trans_state = jnp.concatenate([trans_state, trans_state_end[None]])
    
    # # Common variables throughout the loop
    n_steps, n_obs = y_obs.shape
    mu_hat = jnp.zeros(n_obs)
    wgt_hat = wgt_obs
    # Compute grad and hess of observation model
    hvec = jax.grad(obs_fun, argnums=1)
    Hmat = jax.hessian(obs_fun, argnums=1)

    # reverse pass
    def scan_fun(carry, rev_args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        mean_state_sim = carry["state_sim"]
        logx, logy = carry["logdens"]
        trans_state = rev_args['trans_state']
        mean_state = rev_args['mean_state']
        var_state = rev_args['var_state']
        y_obs = rev_args['y_obs']

        # unconditional mean
        mean_state_sim = trans_state.dot(mean_state_sim) + mean_state
        # kalman predict
        mean_state_pred, var_state_pred = predict(
            mean_state_past=mean_state_filt,
            var_state_past=var_state_filt,
            mean_state=mean_state,
            trans_state=trans_state,
            var_state=var_state
        )
        # kalman update
        # y_obs is None
        def _no_obs():
            mean_state_filt = mean_state_pred
            var_state_filt = var_state_pred
            logyc = 0.0
            return mean_state_filt, var_state_filt, logyc

        # y_obs is not None
        def _obs():
            # current grad and hess
            x_hat = x_fun(mean_state_sim, y_obs, theta)
            Cxhat = wgt_obs.dot(x_hat)
            h_curr = hvec(y_obs, Cxhat, gamma, theta)
            H_curr = Hmat(y_obs, Cxhat, gamma, theta)
            # Compute arguments to setup GMP for xhat
            var_hat = -jnp.linalg.inv(H_curr)
            # observations yhat
            y_hat = Cxhat + var_hat.dot(h_curr)
            # kalman update
            mean_state_filt, var_state_filt = update(
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                x_meas=y_hat,
                mu_meas=mu_hat,
                wgt_meas=wgt_hat,
                var_meas=var_hat
            )
            logyc = obs_fun(y_obs, wgt_obs.dot(mean_state_sim), gamma, theta)
            return mean_state_filt, var_state_filt, logyc

        mean_state_filt, var_state_filt, logyc = jax.lax.cond(jnp.isnan(y_obs).any(), _no_obs, _obs)
        logy = logy + logyc
        logx = logx + jsp.stats.multivariate_normal.logpdf(mean_state_sim, mean_state_sim, var_state)
        # logxc = jsp.stats.multivariate_normal.logpdf(mean_state_sim, mean_state_sim, var_state)

        # output
        carry = {
            "state_filt": (mean_state_filt, var_state_filt),
            "state_sim": mean_state_sim, 
            "logdens": (logx, logy)
        }
        stack = {
            "state_pred": (mean_state_pred, var_state_pred),
            "state_filt": (mean_state_filt, var_state_filt),
            "state_sim": mean_state_sim
        }
        return carry, stack

    # start at N+1 assuming 0 mean and variance
    scan_init = {
        "state_filt" : (jnp.zeros((n_state,)), jnp.zeros((n_state, n_state))),
        "state_sim" : jnp.zeros((n_state,)),
        "logdens": (0.0, 0.0)
    }

    rev_args = {
        "trans_state" : trans_state[1:],
        "mean_state" : mean_state[1:],
        "var_state" : var_state[1:],
        "y_obs": y_obs[1:]
    }

    scan_out, stack_out = jax.lax.scan(scan_fun, scan_init, rev_args, reverse=True)
    # logx = jnp.append(stack_out["logdens"][0], logx)
    logx, logy = scan_out["logdens"]
    logy0 = obs_fun(y_obs[0], wgt_obs.dot(mean_state[0]), gamma, theta)
    logy = logy + logy0
    
    return logx, logy, stack_out

def smoothng(trans_state, state_par):
    r"""
    Computes the logdensity of p(\hat y_{0:N}) which is an approximantion to p(x_{0:N} | y_{0:N}).
    """
    # reverse pass
    mean_state_pred, var_state_pred = state_par["state_pred"]
    mean_state_filt, var_state_filt = state_par["state_filt"]
    mu_uncond = state_par["state_sim"]
    logy_hat = jsp.stats.multivariate_normal.logpdf(mu_uncond[0], mean_state_filt[0], var_state_filt[0])
    # 
    n_steps = len(mean_state_pred)

    # backward pass
    # lax.scan setup
    def scan_fun(logy_hat, smooth_kwargs):
        mean_state_filt = smooth_kwargs['mean_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mean_state_pred = smooth_kwargs['mean_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        trans_state = smooth_kwargs['trans_state']
        mu_prev = smooth_kwargs['mu_prev']
        mu_curr = smooth_kwargs['mu_curr']
        mean_state_sim, var_state_sim = smooth_sim(
            x_state_next = mu_prev,
            mean_state_filt = mean_state_filt,
            var_state_filt = var_state_filt,
            mean_state_pred = mean_state_pred,
            var_state_pred = var_state_pred,
            trans_state = trans_state
        )
        logy_hat += jsp.stats.multivariate_normal.logpdf(mu_curr, mean_state_sim, var_state_sim)
        return logy_hat, None
    # scan arguments
    scan_kwargs = {
        'mean_state_filt': mean_state_filt[1:],
        'var_state_filt': var_state_filt[1:],
        'mean_state_pred': mean_state_pred[:n_steps-1],
        'var_state_pred': var_state_pred[:n_steps-1],
        'mu_curr': mu_uncond[1:],
        'mu_prev': mu_uncond[:n_steps-1],
        'trans_state': trans_state[1:]
    }
    # Note: initial value x0 is assumed to be known, so no need to smooth it
    logy_hat, _= jax.lax.scan(scan_fun, logy_hat, scan_kwargs)
    return logy_hat


def fenrir_filterng(fun, x0, theta, tmin, tmax, n_res,
                    W, trans_state, mean_state, var_state,
                    wgt_obs, y_obs, gamma,
                    obs_fun, x_fun):
    
    r"""
    Fenrir algorithm.

    Args:
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; :math:`W`.
        trans_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_state)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; :math:`R`.
        mu_obs (ndarray(n_obs)): Transition offsets defining the noisy observations.
        wgt_obs (ndarray(n_obs, n_state)): Transition matrix defining the noisy observations; :math:`C`.
        var_obs (ndarray(n_obs, n_obs)): Variance matrix defining the noisy observations; :math:`Omega`.
        y_obs (ndarray(n_steps, n_obs)): Observed data; :math:`y_n`.


    Returns:
        logdens : The logdensity of p(y_{1:N} | x_{1:N}).

    """
    n_obs, n_dim_obs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_dim_obs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)
    # forward pass
    filt_out = forward(
        fun=fun, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, n_steps=n_steps,
        W=W, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state
    )
    mean_state_pred, var_state_pred = filt_out["state_pred"]
    mean_state_filt, var_state_filt = filt_out["state_filt"]

    # backward pass
    trans_state_cond, mean_state_cond, var_state_cond = backward(
        mean_state_filt=mean_state_filt,
        var_state_filt=var_state_filt,
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        trans_state=trans_state
    )

    logx, logy, state_par = reverseng(
        trans_state=trans_state_cond, mean_state=mean_state_cond, 
        var_state=var_state_cond, wgt_obs=wgt_obs,
        y_obs=y_obs, gamma = gamma, theta= theta,
        obs_fun=obs_fun, x_fun = x_fun
    )

    logy_hat = smoothng(trans_state_cond, state_par)
    return logx + logy - logy_hat

