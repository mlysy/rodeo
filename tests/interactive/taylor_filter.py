import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.kalmantv import *
from rodeo.ode import interrogate_rodeo, interrogate_chkrebtii, interrogate_schober, interrogate_tronarp

def _update_2nd(mean_state_pred, var_state_pred, trans_meas, var_meas, fmu, fpmu, fppmu):
    r"""
    Second order Taylor approximation update step.
    """
    var_meas_inv = jnp.linalg.inv(var_meas)
    var_state_inv = jnp.linalg.inv(var_state_pred)
    WsJ = trans_meas - fpmu
    Wmuf = trans_meas.dot(mean_state_pred) - fmu
    Wmuf_inv = Wmuf.T.dot(var_meas_inv)
    b = Wmuf_inv.dot(WsJ)
    c1 = jnp.linalg.multi_dot([
        WsJ.T,
        var_meas_inv,
        WsJ])
    c2 = fppmu.T.dot(Wmuf_inv.T)
    c = c1 - c2
    d = var_state_inv
    mean_state_filt = -jnp.linalg.inv(c+d).dot(b) + mean_state_pred
    hessian = -(c+d)
    var_state_filt = -jnp.linalg.inv(hessian)
    return mean_state_filt, var_state_filt

def _solve_filter(key, fun,  W,  x0, theta,
                  tmin, tmax, n_steps,
                  trans_state, mean_state, var_state):
    r"""
    Forward pass of the ODE solver.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        interrogate (function): Function defining the interrogation method.

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
    almost_zero = 1e-8
    var_meas = jax.vmap(lambda wm, vsp:
                        jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
        W, var_state
    )*1e-2
    # lax.scan setup
    # scan function
    def scan_fun(carry, t):
        mean_state_filt, var_state_filt = carry["state_filt"]
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
        t_n = tmin + (tmax-tmin)*(t+1)/n_steps
        fmu = fun(mean_state_pred, t_n, theta)
        fpmu = jax.jacfwd(fun)(mean_state_pred, t_n, theta)
        fppmu = jax.jacfwd(jax.jacrev(fun))(mean_state_pred, t_n, theta)
        # kalman update
        mean_state_next, var_state_next = jax.vmap(lambda b:
            _update_2nd(
                mean_state_pred=mean_state_pred[b],
                var_state_pred=var_state_pred[b],
                trans_meas=W[b],
                var_meas=var_meas[b],
                fmu=fmu[b],
                fpmu=fpmu[b, :, b], 
                fppmu=fppmu[b, :, b, :, b]
            )
        )(jnp.arange(n_block))
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

def solve_mv(key, fun, W, x0, theta,
             tmin, tmax, n_steps,
             trans_state, mean_state, var_state):
    r"""
    Mean and variance of the stochastic ODE solver.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mean_state.shape
    # forward pass
    filt_out = _solve_filter(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_steps=n_steps, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state
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

def _update_2nd_obs(mean_state_pred, var_state_pred, trans_meas, var_meas, fmu, fpmu, fppmu, gpmu, gppmu):
    r"""
    Second order Taylor approximation with observations update step.
    """
    var_meas_inv = jnp.linalg.inv(var_meas)
    var_state_inv = jnp.linalg.inv(var_state_pred)
    WsJ = trans_meas - fpmu
    Wmuf = trans_meas.dot(mean_state_pred) - fmu
    Wmuf_inv = Wmuf.T.dot(var_meas_inv)
    b = Wmuf_inv.dot(WsJ)
    c1 = jnp.linalg.multi_dot([
        WsJ.T,
        var_meas_inv,
        WsJ])
    c2 = fppmu.T.dot(Wmuf_inv.T)
    c = c1 - c2
    d = var_state_inv
    mean_state_filt = -jnp.linalg.inv(c + d - gppmu).dot(b - gpmu) + mean_state_pred
    hessian = -(c+d) + gppmu
    var_state_filt = -jnp.linalg.inv(hessian)
    return mean_state_filt, var_state_filt

def _solve_filter_obs(key, fun,  W,  x0, theta,
                  tmin, tmax, n_res,
                  trans_state, mean_state, var_state,
                  fun_obs, y_obs):
    r"""
    Forward pass of the ODE solver.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Reshaping y_obs to be in blocks 
    n_obs, n_block, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_block, n_bobs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)

    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape

    # arguments for kalman_filter and kalman_smooth
    mean_state_init = x0
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))
    var_meas = jax.vmap(lambda wm, vsp:
                        jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
        W, var_state
    )*1e-2
    # lax.scan setup
    # scan function
    def scan_fun(carry, args):
        mean_state_filt, var_state_filt = carry["state_filt"]
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
        t_n = tmin + (tmax-tmin)*(t+1)/n_steps
        fmu = fun(mean_state_pred, t_n, theta)
        fpmu = jax.jacfwd(fun)(mean_state_pred, t_n, theta)
        fppmu = jax.jacfwd(jax.jacrev(fun))(mean_state_pred, t_n, theta)
        
        # observation
        def zy_update():
            gpmu = jax.jacfwd(fun_obs)(mean_state_pred, y_curr)
            gppmu = jax.jacfwd(jax.jacrev(fun_obs))(mean_state_pred, y_curr)
            mean_state_next, var_state_next = jax.vmap(lambda b:
                _update_2nd_obs(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    trans_meas=W[b],
                    var_meas=var_meas[b],
                    fmu=fmu[b],
                    fpmu=fpmu[b, :, b], 
                    fppmu=fppmu[b, :, b, :, b],
                    gpmu=gpmu[b],
                    gppmu=gppmu[b, :, b]
                )
            )(jnp.arange(n_block))
            return mean_state_next, var_state_next
        
        # no observation
        def z_update():
            mean_state_next, var_state_next = jax.vmap(lambda b:
                _update_2nd(
                    mean_state_pred=mean_state_pred[b],
                    var_state_pred=var_state_pred[b],
                    trans_meas=W[b],
                    var_meas=var_meas[b],
                    fmu=fmu[b],
                    fpmu=fpmu[b, :, b], 
                    fppmu=fppmu[b, :, b, :, b]
                )
            )(jnp.arange(n_block))
            return mean_state_next, var_state_next
        

        mean_state_next, var_state_next = jax.lax.cond(jnp.isnan(y_curr).any(), z_update, zy_update)
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
        "state_filt": (mean_state_init, var_state_init)
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


def solve_mv_obs(key, fun, W, x0, theta,
             tmin, tmax, n_res,
             trans_state, mean_state, var_state,
             fun_obs, y_obs):
    r"""
    Mean and variance of the stochastic ODE solver.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = b/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps+1, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mean_state.shape
    # forward pass
    filt_out = _solve_filter_obs(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_res=n_res, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        fun_obs=fun_obs, y_obs=y_obs
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



def _solve_filter_obs2(key, fun,  W,  x0, theta,
                  tmin, tmax, n_res,
                  trans_state, mean_state, var_state,
                  fun_obs, trans_obs, y_obs, interrogate):
    r"""
    Forward pass of the ODE solver.

    Args:
        key (PRNGKey): PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior; :math:`W`.
        x0 (ndarray(n_block, n_bstate)): Initial value of the state variable :math:`X_t` at time :math:`t = a`.
        theta (ndarray(n_theta)): Parameters in the ODE function.
        tmin (float): First time point of the time interval to be evaluated; :math:`a`.
        tmax (float): Last time point of the time interval to be evaluated; :math:`b`.
        n_steps (int): Number of discretization points (:math:`N`) of the time interval that is evaluated, such that discretization timestep is :math:`dt = (b-a)/N`.
        trans_state (ndarray(n_block, n_bstate, n_bstate)): Transition matrix defining the solution prior; :math:`Q`.
        mean_state (ndarray(n_block, n_bstate)): Transition_offsets defining the solution prior; :math:`c`.
        var_state (ndarray(n_block, n_bstate, n_bstate)): Variance matrix defining the solution prior; :math:`R`.
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps+1, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps+1, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Reshaping y_obs to be in blocks 
    n_obs, n_block, n_bobs = y_obs.shape
    n_steps = (n_obs-1)*n_res
    y_res = jnp.ones((n_steps+1, n_block, n_bobs))*jnp.nan
    y_obs = y_res.at[::n_res].set(y_obs)

    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape

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
        
        # observation
        def zy_update():
            Cmu = jax.vmap(lambda b: trans_obs[b].dot(mean_state_pred[b]))(jnp.arange(n_block))
            gpmu = jax.jacfwd(fun_obs)(Cmu, y_curr)
            gppmu = jax.jacfwd(jax.jacrev(fun_obs))(Cmu, y_curr)
            var_obs = jax.vmap(lambda b: -jnp.linalg.inv(gppmu[b, :, b]))(jnp.arange(n_block))
            y_new = jax.vmap(lambda b: Cmu[b] + var_obs[b].dot(gpmu[b]))(jnp.arange(n_block))
            trans_meas_obs = jnp.concatenate([trans_meas, trans_obs], axis=1)
            mean_meas_obs = jnp.concatenate([mean_meas, mean_obs], axis=1)
            # var_meas_obs = jnp.concatenate([var_meas, var_obs], axis=1)
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
        
        # no observation
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

def solve_mv_obs2(key, fun, W, x0, theta,
             tmin, tmax, n_res,
             trans_state, mean_state, var_state,
             fun_obs, trans_obs, y_obs, 
             interrogate=interrogate_tronarp):
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
    n_block, n_bstate = mean_state.shape
    # forward pass
    filt_out = _solve_filter_obs2(
        key=key,
        fun=fun, W=W, x0=x0, theta=theta,
        tmin=tmin, tmax=tmax, 
        n_res=n_res, trans_state=trans_state,
        mean_state=mean_state, var_state=var_state,
        fun_obs=fun_obs, trans_obs=trans_obs, y_obs=y_obs,
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
