r"""
Time-varying Kalman filtering and smoothing algorithms.

The Gaussian state space model underlying the algorithms is

.. math::

    x_n = c_n + Q_n x_{n-1} + R_n^{1/2} \epsilon_n

    y_n = d_n + W_n x_n + V_n^{1/2} \eta_n,

where :math:`\epsilon_n \stackrel{\text{iid}}{\sim} \operatorname{Normal}(0, I_p)` and independently :math:`\eta_n \stackrel{\text{iid}}{\sim} \operatorname{Normal}(0, I_q)`.  At each time :math:`n`, only :math:`y_n` is observed.  The Kalman filtering and smoothing algorithms efficiently calculate quantities of the form :math:`\theta_{m|n} = (\mu_{m|n}, \Sigma_{m|n})`, where

.. math::

    \mu_{m|n} = E[x_m \mid y_{0:n}]

    \Sigma_{m|n} = \text{var}(x_m \mid y_{0:n}),

for different combinations of :math:`m` and :math:`n`.

"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp


# --- helper functions ---------------------------------------------------
def _solveV(V, B):
    r"""
    Computes :math:`X = V^{-1}B` where V is a variance matrix.

    Args:
        V (ndarray(n_dim1, n_dim1)): Variance matrix V in :math:`X = V^{-1}B`.
        B (ndarray(n_dim1, n_dim2)): Matrix B in :math:`X = V^{-1}B`.

    Returns:
        (ndarray(n_dim1, n_dim2)): Matrix X in :math:`X = V^{-1}B`

    """
    # L, low = jsp.linalg.cho_factor(V)
    # return jsp.linalg.cho_solve((L, low), B)
    return jsp.linalg.solve(V, B)

# --- core functions -----------------------------------------------------


def predict(mean_state_past, 
            var_state_past,
            mean_state, 
            wgt_state,
            var_state, 
            *args, **kwargs):
    r"""
    Perform one prediction step of the Kalman filter.

    Calculates :math:`\theta_{n|n-1}` from :math:`\theta_{n-1|n-1}`.

    Args:
        mean_state_past (ndarray(n_state)): Mean estimate for state at time n-1 given observations from times [0...n-1]; :math:`\mu_{n-1|n-1}`.
        var_state_past (ndarray(n_state, n_state)): Covariance of estimate for state at time n-1 given observations from times [0...n-1]; :math:`\Sigma_{n-1|n-1}`.
        mean_state (ndarray(n_state)): Transition offsets defining the solution prior; denoted by :math:`c_n`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q_n`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R_n`.
        *args (args): Extra arguments for different Kalman equations.
        **kwargs (kwargs): Extra keyword arguments for different Kalman equations.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        - **var_state_pred** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.

    """
    mean_state_pred = wgt_state.dot(mean_state_past) + mean_state
    var_state_pred = jnp.linalg.multi_dot(
        [wgt_state, var_state_past, wgt_state.T]) + var_state
    return mean_state_pred, var_state_pred


def update(mean_state_pred,
           var_state_pred,
           x_meas,
           mean_meas,
           wgt_meas,
           var_meas,
           *args, **kwargs):
    r"""
    Perform one update step of the Kalman filter.

    Calculates :math:`\theta_{n|n}` from :math:`\theta_{n|n-1}`.

    Args:
        mean_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        x_meas (ndarray(n_meas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mean_meas (ndarray(n_meas)): Transition offsets defining the measure prior; denoted by :math:`d_n`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; denoted by :math:`W_n`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`V_n`.
        *args (args): Extra arguments for different Kalman equations.
        **kwargs (kwargs): Extra keyword arguments for different Kalman equations.

    Returns:
        (tuple):
        - **mean_state_filt** (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n}`.
        - **var_state_filt** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.

    """
    mean_meas_pred = wgt_meas.dot(mean_state_pred) + mean_meas
    var_meas_state_pred = wgt_meas.dot(var_state_pred)
    var_meas_meas_pred = jnp.linalg.multi_dot(
        [wgt_meas, var_state_pred, wgt_meas.T]) + var_meas
    var_state_meas_pred = var_state_pred.dot(wgt_meas.T)
    var_state_temp = _solveV(var_meas_meas_pred, var_state_meas_pred.T).T
    mean_state_filt = mean_state_pred + \
        var_state_temp.dot(x_meas - mean_meas_pred)
    var_state_filt = var_state_pred - \
        var_state_temp.dot(var_meas_state_pred)
    return mean_state_filt, var_state_filt


def filter(mean_state_past,
           var_state_past,
           mean_state,
           wgt_state,
           var_state,
           x_meas,
           mean_meas,
           wgt_meas,
           var_meas,
           *args, **kwargs):
    r"""
    Perform one step of the Kalman filter.

    Combines :func:`kalmantv.predict` and :func:`kalmantv.update` steps to get :math:`\theta_{n|n}` from :math:`\theta_{n-1|n-1}`.

    Args:
        mean_state_past (ndarray(n_state)): Mean estimate for state at time n-1 given observations from times [0...n-1]; :math:`\mu_{n-1|n-1}`.
        var_state_past (ndarray(n_state, n_state)): Covariance of estimate for state at time n-1 given observations from times [0...n-1]; :math:`\Sigma_{n-1|n-1}`.
        mean_state (ndarray(n_state)): Transition offsets defining the solution prior; denoted by :math:`c_n`.
        wgt_state (ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q_n`.
        var_state (ndarray(n_state, n_state)): Variance matrix defining the solution prior; denoted by :math:`R_n`.
        x_meas (ndarray(n_meas)): Interrogated measure vector from `x_state`; :math:`y_n`.
        mean_meas (ndarray(n_meas)): Transition offsets defining the measure prior; denoted by :math:`d_n`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; denoted by :math:`W_n`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`V_n`.
        *args (args): Extra arguments for different Kalman equations.
        **kwargs (kwargs): Extra keyword arguments for different Kalman equations.


    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        - **var_state_pred** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        - **mean_state_filt** (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n}`.
        - **var_state_filt** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.

    """
    mean_state_pred, var_state_pred = predict(
        mean_state_past=mean_state_past,
        var_state_past=var_state_past,
        mean_state=mean_state,
        wgt_state=wgt_state,
        var_state=var_state
    )
    mean_state_filt, var_state_filt = update(
        mean_state_pred=mean_state_pred,
        var_state_pred=var_state_pred,
        x_meas=x_meas,
        mean_meas=mean_meas,
        wgt_meas=wgt_meas,
        var_meas=var_meas
    )
    return mean_state_pred, var_state_pred, mean_state_filt, var_state_filt


def _smooth(var_state_filt, var_state_pred, wgt_state, *args, **kwargs):
    r"""
    Common part of :func:`kalmantv.smooth_sim` and :func:`kalmantv.smooth_mv`.

    Args:
        var_state_filt(ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...n]; denoted by :math:`\Sigma_{n | n}`.
        var_state_pred(ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...n-1]; denoted by :math:`\Sigma_{n | n-1}`.
        wgt_state(ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q`.
        *args (args): Extra arguments for different Kalman equations.
        **kwargs (kwargs): Extra keyword arguments for different Kalman equations.


    Returns:
        (tuple):
        - **var_state_temp** (ndarray(n_state, n_state)): Tempory variance calculation used by :func:`kalmantv.smooth_sim`.
        - **var_state_temp_tilde** (ndarray(n_state, n_state)): Tempory variance calculation used by :func:`kalmantv.smooth_sim` and :func:`kalmantv.smooth_mv`.
    """
    var_state_temp = var_state_filt.dot(wgt_state.T)
    var_state_temp_tilde = _solveV(var_state_pred, var_state_temp.T).T
    return var_state_temp, var_state_temp_tilde


def smooth_mv(mean_state_next,
              var_state_next,
              mean_state_filt,
              var_state_filt,
              mean_state_pred,
              var_state_pred,
              wgt_state,
              *args, **kwargs):
    r"""
    Perform one step of the Kalman mean/variance smoother.

    Calculates :math:`\theta_{n|N}` from :math:`\theta_{n+1|N}`, :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`.

    Args:
        mean_state_next(ndarray(n_state)): Mean estimate for state at time n+1 given observations from times[0...N]; denoted by :math:`\mu_{n+1 | N}`.
        var_state_next(ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given observations from times[0...N]; denoted by :math:`\Sigma_{n+1 | N}`.
        mean_state_filt(ndarray(n_state)): Mean estimate for state at time n given observations from times[0...n]; denoted by :math:`\mu_{n | n}`.
        var_state_filt(ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...n]; denoted by :math:`\Sigma_{n | n}`.
        mean_state_pred(ndarray(n_state)): Mean estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\mu_{n+1 | n}`.
        var_state_pred(ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\Sigma_{n+1 | n}`.
        wgt_state(ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q_{n+1}`.
        *args (args): Extra arguments for different Kalman equations.
        **kwargs (kwargs): Extra keyword arguments for different Kalman equations.


    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_state)): Mean estimate for state at time n given observations from times[0...N]; denoted by :math:`\mu_{n | N}`.
        - **var_state_smooth** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...N]; denoted by :math:`\Sigma_{n | N}`.

    """
    var_state_temp, var_state_temp_tilde = _smooth(
        var_state_filt, var_state_pred, wgt_state
    )
    mean_state_smooth = mean_state_filt + \
        var_state_temp_tilde.dot(mean_state_next - mean_state_pred)
    var_state_smooth = var_state_filt + jnp.linalg.multi_dot(
        [var_state_temp_tilde, (var_state_next - var_state_pred), var_state_temp_tilde.T])
    return mean_state_smooth, var_state_smooth


def smooth_sim(x_state_next,
               mean_state_filt,
               var_state_filt,
               mean_state_pred,
               var_state_pred,
               wgt_state,
               *args, **kwargs):
    r"""
    Perform one step of the Kalman sampling smoother.

    Calculates :math:`\tilde \theta_{n|N}` from :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`, i.e., :math:`x_{n | N} | x_{n+1 | N} \sim \operatorname{Normal}(\tilde \mu_{n | N}, \tilde \Sigma_{n | N})`.

    Args:
        x_state_next(ndarray(n_state)): Simulated state at time n+1 given observations from times[0...N]; denoted by :math:`x_{n+1 | N}`.
        mean_state_filt(ndarray(n_state)): Mean estimate for state at time n given observations from times[0...n]; denoted by :math:`\mu_{n | n}`.
        var_state_filt(ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...n]; denoted by :math:`\Sigma_{n | n}`.
        mean_state_pred(ndarray(n_state)): Mean estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\mu_{n+1 | n}`.
        var_state_pred(ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\Sigma_{n+1 | n}`.
        wgt_state(ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q_{n+1}`.
        *args (args): Extra arguments for different Kalman equations.
        **kwargs (kwargs): Extra keyword arguments for different Kalman equations.


    Returns:
        (tuple):
        - **mean_state_sim** (ndarray(n_state)): Mean estimate for state at time n given observations from times[0...N] and :math:`x_{n+1 | N}`; denoted by :math:`\tilde \mu_{n | N}`.
        - **var_state_sim** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...N] and :math:`x_{n+1 | N}`; denoted by :math;`\tilde \Sigma_{n | N}`.

    """
    var_state_temp, var_state_temp_tilde = _smooth(
        var_state_filt, var_state_pred, wgt_state
    )
    mean_state_sim = mean_state_filt + \
        var_state_temp_tilde.dot(x_state_next - mean_state_pred)
    var_state_sim = var_state_filt - \
        var_state_temp_tilde.dot(var_state_temp.T)
    return mean_state_sim, var_state_sim


def smooth(x_state_next,
           mean_state_next,
           var_state_next,
           mean_state_filt,
           var_state_filt,
           mean_state_pred,
           var_state_pred,
           wgt_state,
           *args, **kwargs):
    r"""
    Perform one step of both Kalman mean/variance and sampling smoothers.

    Combines :func:`kalmantv.smooth_mv` and :func:`kalmantv.smooth_sim` steps to get the mean and variance of :math:`x_{n|N}` and :math:`\theta_{n|N}` from :math:`\theta_{n+1|N}`, :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`.

    Args:
        x_state_next(ndarray(n_state)): Simulated state at time n+1 given observations from times[0...N]; denoted by :math:`x_{n+1 | N}`.
        mean_state_next(ndarray(n_state)): Mean estimate for state at time n+1 given observations from times[0...N]; denoted by :math:`\mu_{n+1 | N}`.
        var_state_next(ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given observations from times[0...N]; denoted by :math:`\Sigma_{n+1 | N}`.
        mean_state_filt(ndarray(n_state)): Mean estimate for state at time n given observations from times[0...n]; denoted by :math:`\mu_{n | n}`.
        var_state_filt(ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...n]; denoted by :math:`\Sigma_{n | n}`.
        mean_state_pred(ndarray(n_state)): Mean estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\mu_{n+1 | n}`.
        var_state_pred(ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\Sigma_{n | n}`.
        wgt_state(ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q_{n+1}`.
        *args (args): Extra arguments for different Kalman equations.
        **kwargs (kwargs): Extra keyword arguments for different Kalman equations.


    Returns:
        (tuple):
        - **mean_state_sim** (ndarray(n_state)): Mean estimate for state at time n given observations from times[0...N] and :math:`x_{n+1 | N}`; denoted by :math:`\tilde \mu_{n | N}`.
        - **var_state_sim** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...N] and :math:`x_{n+1 | N}`; denoted by :math;`\tilde \Sigma_{n | N}`.
        - **mean_state_smooth** (ndarray(n_state)): Mean estimate for state at time n given observations from times[0...N]; denoted by :math:`\mu_{n | N}`.
        - **var_state_smooth** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...N]; denoted by :math:`\Sigma_{n | N}`.

    """
    var_state_temp, var_state_temp_tilde = _smooth(
        var_state_filt, var_state_pred, wgt_state
    )
    mean_state_temp = jnp.concatenate([x_state_next[None],
                                       mean_state_next[None]])
    mean_state_temp = mean_state_filt + \
        var_state_temp_tilde.dot((mean_state_temp - mean_state_pred).T).T
    mean_state_sim = mean_state_temp[0]
    var_state_sim = var_state_filt - \
        var_state_temp_tilde.dot(var_state_temp.T)
    mean_state_smooth = mean_state_temp[1]
    var_state_smooth = var_state_filt + \
        jnp.linalg.multi_dot(
            [var_state_temp_tilde, (var_state_next - var_state_pred),
             var_state_temp_tilde.T])
    return mean_state_sim, var_state_sim, mean_state_smooth, var_state_smooth,


def forecast(mean_state_pred,
             var_state_pred,
             mean_meas,
             wgt_meas,
             var_meas,
             *args, **kwargs):
    r"""
    Forecasts the mean and variance of the measurement at time step n given observations from times[0...n-1].

    Args:
        mean_state_pred(ndarray(n_state)): Mean estimate for state at time n given observations from times[0...n-1]; denoted by :math:`\mu_{n | n-1}`.
        var_state_pred(ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...n-1]; denoted by :math:`\Sigma_{n | n-1}`.
        mean_meas (ndarray(n_meas)): Transition offsets defining the measure prior; denoted by :math:`d_n`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior; denoted by :math:`W_n`.
        var_meas (ndarray(n_meas, n_meas)): Variance matrix defining the measure prior; denoted by :math:`V_n`.
        *args (args): Extra arguments for different Kalman equations.
        **kwargs (kwargs): Extra keyword arguments for different Kalman equations.


    Returns:
        (tuple):
        - **mean_fore** (ndarray(n_meas)): Mean estimate for measurement at n given observations from [0...n-1].
        - **var_fore** (ndarray(n_meas, n_meas)): Covariance of estimate for state at time n given observations from times[0...n-1].
    """
    mean_fore = wgt_meas.dot(mean_state_pred) + mean_meas
    var_fore = jnp.linalg.multi_dot(
        [wgt_meas, var_state_pred, wgt_meas.T]) + var_meas
    return mean_fore, var_fore


def smooth_cond(mean_state_filt,
                var_state_filt,
                mean_state_pred,
                var_state_pred,
                wgt_state,
                *args, **kwargs):
    r"""
    Perform one step of the Kalman sampling smoother conditional.

    Calculates :math:`A_{n|N}`, :math:`b_{n|N}`, :math:`C_{n|N}` from :math:`\theta_{n|n}`, and :math:`\theta_{n+1|n}`.

    Args:
        mean_state_filt(ndarray(n_state)): Mean estimate for state at time n given observations from times[0...n]; denoted by :math:`\mu_{n | n}`.
        var_state_filt(ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times[0...n]; denoted by :math:`\Sigma_{n | n}`.
        mean_state_pred(ndarray(n_state)): Mean estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\mu_{n+1 | n}`.
        var_state_pred(ndarray(n_state, n_state)): Covariance of estimate for state at time n+1 given observations from times[0...n]; denoted by :math:`\Sigma_{n+1 | n}`.
        wgt_state(ndarray(n_state, n_state)): Transition matrix defining the solution prior; denoted by :math:`Q_{n+1}`.
        *args (args): Extra arguments for different Kalman equations.
        **kwargs (kwargs): Extra keyword arguments for different Kalman equations.


    Returns:
        - **wgt_state_cond** (ndarray(n_state, n_state)): Transition of smooth conditional at time n given observations from times[0...N]; :math:`A_{n|N}`.
        - **mean_state_cond** (ndarray(n_state)): Offset of smooth conditional at time n given observations from times[0...N]; :math:`b_{n|N}`.
        - *var_state_cond** (ndarray(n_state, n_state)): Variance of smooth conditional at time n given observations from times[0...N]; :math:`C_{n|N}`.

    """
    var_state_temp, wgt_state_cond = _smooth(
        var_state_filt, var_state_pred, wgt_state
    )
    mean_state_cond = mean_state_filt - wgt_state_cond.dot(mean_state_pred)
    var_state_cond = var_state_filt - wgt_state_cond.dot(var_state_temp.T)
    return wgt_state_cond, mean_state_cond, var_state_cond
