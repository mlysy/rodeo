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

def kalman_zero(fun, W, t, theta,
                mean_state_pred, var_state_pred):
    
    n_block = mean_state_pred.shape[0]
    var_meas = jax.vmap(lambda wm, vsp:
                        jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
        W, var_state_pred
    )
    # var_meas = jnp.zeros((n_block, n_bmeas, n_bmeas))
    mean_meas = -fun(mean_state_pred, t, theta)
    return W, mean_meas, var_meas

# use interrogations first then observations
def forward(fun, W, x0, theta,
            tmin, tmax, n_steps,
            trans_state, mean_state, var_state):
    r"""
    Forward pass of the Fenrir algorithm.

    Args:
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
        trans_meas, mean_meas, var_meas = kalman_zero(
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
        - *var_state_cond** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance of smooth conditional at time t given observations from times [0...T]; :math:`V_{n|N}`.

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
    # Slice these arrays so they are aligned.
    # More precisely, for time step t, want filt[t], pred[t+1]
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

def backward(trans_state, mean_state, var_state,
             trans_obs, mean_obs, var_obs, y_obs):
    
    r"""
    Backward pass of Fenrir algorithm. Uses observations.

    Args:
        trans_state (ndarray(n_steps, n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`A^S`.
        mean_state (ndarray(n_steps+1, n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`b^S`.
        var_state (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`V^S`.
        mean_obs (ndarray(n_block, n_bmeas)): Transition offsets defining the noisy observations.
        trans_obs (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the noisy observations; :math:`D`.
        var_obs (ndarray(n_block, n_bmeas, n_bmeas)): Variance matrix defining the noisy observations; :math:`Omega`.
        y_obs (ndarray(n_steps, n_block, n_bmeas)): Observed data; :math:`y_n`.

    Returns:
        logdens : The logdensity of p(Y_{0:N}).

    """
    # Add point to beginning of state variable for simpler loop
    n_tot, n_block, n_bstate = mean_state.shape
    trans_state_end = jnp.zeros(trans_state.shape[1:])
    trans_state = jnp.concatenate([trans_state, trans_state_end[None]])

    # reverse pass
    def scan_fun(carry, back_args):
        mean_state_filt, var_state_filt = carry["state_filt"]
        logdens = carry["logdens"]
        trans_state = back_args['trans_state']
        mean_state = back_args['mean_state']
        var_state = back_args['var_state']
        y_obs = back_args['y_obs']

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
            # logp = 0.0
            logp = jnp.sum(jax.vmap(lambda b:
                jsp.stats.multivariate_normal.logpdf(y_obs[b], mean=mean_state_fore[b], cov=var_state_fore[b])
            )(jnp.arange(n_block)))
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
            return mean_state_filt, var_state_filt, logp

        mean_state_filt, var_state_filt, logp = jax.lax.cond(jnp.isnan(y_obs).any(), _no_obs, _obs)
        logdens += logp

        # output
        carry = {
            "state_filt": (mean_state_filt, var_state_filt),
            "logdens" : logdens
        }
        # stack = {
        #     "state_pred": (mean_state_pred, var_state_pred),
        #     "state_filt": (mean_state_filt, var_state_filt)
        # }
        return carry, carry

    # start at N+1 assuming 0 mean and variance
    scan_init = {
        "state_filt" : (jnp.zeros((n_block, n_bstate,)), jnp.zeros((n_block, n_bstate, n_bstate))),
        "logdens" : 0.0
    }
    back_args = {
        "trans_state" : trans_state,
        "mean_state" : mean_state,
        "var_state" : var_state,
        "y_obs": y_obs
    }

    scan_out, _ = jax.lax.scan(scan_fun, scan_init, back_args, reverse=True)
    return scan_out["logdens"]
    
def fenrir(fun, W, x0, theta, tmin, tmax, n_res,
            trans_state, mean_state, var_state,
            trans_obs, mean_obs, var_obs, y_obs):
    
    r"""
    Fenrir algorithm.

    Args:
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        mean_obs (ndarray(n_block, n_bmeas)): Transition offsets defining the noisy observations.
        trans_obs (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the noisy observations; :math:`D`.
        var_obs (ndarray(n_block, n_bmeas, n_bmeas)): Variance matrix defining the noisy observations; :math:`Omega`.
        y_obs (ndarray(n_steps, n_meas)): Observed data; :math:`y_n`.


    Returns:
        logdens : The logdensity of p(y_{0:N}).

    """
    n_obs, n_dim_obs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_dim_obs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)
    y_obs = jnp.expand_dims(y_obs, -1)
    # forward pass
    filt_out = forward(
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, n_steps=n_steps,
        trans_state=trans_state,
        mean_state=mean_state, var_state=var_state
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
    logdens = backward(
        trans_state=trans_state_cond, mean_state=mean_state_cond, 
        var_state=var_state_cond, trans_obs=trans_obs,
        mean_obs=mean_obs, var_obs=var_obs, y_obs=y_obs
    )
    # state_par = reverse(
    #     trans_state=trans_state_cond, mean_state=mean_state_cond, 
    #     var_state=var_state_cond, trans_obs=trans_obs,
    #     mean_obs=mean_obs, var_obs=var_obs, y_obs=y_obs
    # )
    # mean_state_smooth, var_state_smooth = smooth(trans_state_cond, state_par)
    return logdens
