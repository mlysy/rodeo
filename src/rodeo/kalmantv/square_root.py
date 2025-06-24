r"""
Time-varying square-root Kalman filtering and smoothing algorithms.

The Gaussian state space model underlying the algorithms is

.. math::

    x_n = c_n + Q_n x_{n-1} + R_n^{1/2} \epsilon_n

    y_n = d_n + W_n x_n + V_n^{1/2} \eta_n,

where :math:`\epsilon_n \stackrel{\text{iid}}{\sim} \operatorname{Normal}(0, I_p)` and independently :math:`\eta_n \stackrel{\text{iid}}{\sim} \operatorname{Normal}(0, I_q)`.  At each time :math:`n`, only :math:`y_n` is observed.  The suqare-root Kalman filtering and smoothing algorithms efficiently calculate quantities of the form :math:`\theta_{m|n} = (\mu_{m|n}, \Gamma_{m|n})`, where

.. math::

    \mu_{m|n} = E[x_m \mid y_{0:n}]

    \Gamma_{m|n} = \text{var}(x_m \mid y_{0:n})^{1/2},

for different combinations of :math:`m` and :math:`n`.

"""
import jax
import jax.numpy as jnp
from rodeo.utils import add_sqrt


# --- core functions -----------------------------------------------------------

def predict(mean_state_past, 
            var_state_past,
            mean_state, 
            wgt_state,
            var_state,
            *args, **kwargs):
    r"""
    Performs one prediction step of the square-root Kalman filter.

    Calculates the mean and square-root variance of :math:`p(X_n | Z_{0:n-1})` from :math:`p(X_{n-1} | Z_{0:n-1})`.

    Args:
        mean_state_past (ndarray(n_state)): State mean at time :math:`t = n-1` given observations at times :math:`t = 0, \dots, n-1`.
        var_state_past (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n-1` given observations at times :math:`t = 0, \dots, n-1`.
        mean_state (ndarray(n_state)): State equation offset at time :math:`t = n`.
        wgt_state (ndarray(n_state, n_state)): State transition matrix at time :math:`t = n`.
        var_state (ndarray(n_state, n_state)): State square-root variance matrix at time :math:`t = n`.
        args (Optional[pytree]): Additional positional arguments for Kalman functions.
        kwargs (Optional[pytree]): Additional keyword arguments for Kalman functions.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
        - **var_state_pred** (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
    """

    mean_state_pred = wgt_state.dot(mean_state_past) + mean_state  
    var_state_pred = add_sqrt(wgt_state.dot(var_state_past), var_state)
    return mean_state_pred, var_state_pred


def update(mean_state_pred,
           var_state_pred,
           x_meas,
           mean_meas,
           wgt_meas,
           var_meas,
           *args, **kwargs):
    r"""
    Performs one update step of the square-root Kalman filter.

    Calculates the mean and square-root variance of :math:`p(X_n | Z_{0:n})` from :math:`p(X_n | Z_{0:n-1})`.

    Args:
        mean_state_pred (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
        var_state_pred (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
        x_meas (ndarray(n_meas)): Interrogated measure vector from :math:`x_{\text{state}}` at time :math:`t = n`.
        mean_meas (ndarray(n_meas)): Measurement equation offset at time :math:`t = n`.
        wgt_meas (ndarray(n_meas, n_state)): Measurement transition matrix at time :math:`t = n`.
        var_meas (ndarray(n_meas, n_meas)): Measurement square-root variance matrix at time :math:`t = n`.
        args (Optional[pytree]): Additional positional arguments for Kalman functions.
        kwargs (Optional[pytree]): Additional keyword arguments for Kalman functions.

    Returns:
        (tuple):
        - **mean_state_filt** (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        - **var_state_filt** (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
    """

    mean_meas_pred = wgt_meas.dot(mean_state_pred) + mean_meas
    var_meas_meas_pred = add_sqrt(wgt_meas.dot(var_state_pred),var_meas)
    # variance_state_pred = var_state_pred.dot(var_state_pred.T)
    intermediate = jax.scipy.linalg.solve_triangular(var_meas_meas_pred, wgt_meas, lower=True)
    # intermediate = intermediate.dot(variance_state_pred)
    intermediate = jnp.linalg.multi_dot([intermediate, var_state_pred, var_state_pred.T])
    var_state_temp = jax.scipy.linalg.solve_triangular(var_meas_meas_pred.T, intermediate, lower=False).T   
    mean_state_filt = mean_state_pred + \
        var_state_temp.dot(x_meas - mean_meas_pred)
    var_state_filt = add_sqrt(var_state_pred - (var_state_temp.dot(wgt_meas)).dot(var_state_pred), 
                            var_state_temp.dot(var_meas))
    
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
    Performs one step of the square-root Kalman filter.

    Combines the :func:`predict` and :func:`update` steps at time :math:`t = n`.

    Args:
        mean_state_past (ndarray(n_state)): State mean at time :math:`t = n-1` given observations at times :math:`t = 0, \dots, n-1`.
        var_state_past (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n-1` given observations at times :math:`t = 0, \dots, n-1`.
        mean_state (ndarray(n_state)): State equation offset at time :math:`t = n`.
        wgt_state (ndarray(n_state, n_state)): State transition matrix at time :math:`t = n`.
        var_state (ndarray(n_state, n_state)): State square-root variance matrix at time :math:`t = n`.
        x_meas (ndarray(n_state)): Interrogated measure vector from :math:`x_{\text{state}}` at time :math:`t = n`.
        mean_meas (ndarray(n_state)): Transition offsets defining the measure prior at time :math:`t = n`.
        wgt_meas (ndarray(n_meas, n_state)): Transition matrix defining the measure prior at time :math:`t = n`.
        var_meas (ndarray(n_meas, n_meas)): Measurement square-root variance matrix at time :math:`t = n`.
        args (Optional[pytree]): Additional positional arguments for Kalman functions.
        kwargs (Optional[pytree]): Additional keyword arguments for Kalman functions.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
        - **var_state_pred** (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
        - **mean_state_filt** (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        - **var_state_filt** (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
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


def _smooth(var_state_filt, var_state_pred, wgt_state):
    r"""
    Common part of :func:`smooth_sim` and :func:`smooth_mv`.

    Args:
        var_state_filt (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        var_state_pred (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
        wgt_state (ndarray(n_state, n_state)): State transition matrix at time :math:`t = n+1`.

    Returns:
        (ndarray(n_state, n_state)): Temporary square-root variance calculation used by :func:`smooth_sim` and :func:`smooth_mv`.
    """

    variance_state_filt = var_state_filt.dot(var_state_filt.T)
    intermediate = jax.scipy.linalg.solve_triangular(var_state_pred, wgt_state, lower=True)
    intermediate = intermediate.dot(variance_state_filt)
    var_state_temp_tilde = jax.scipy.linalg.solve_triangular(var_state_pred.T, intermediate, lower=False).T
    return var_state_temp_tilde


def smooth_mv(mean_state_next,
              var_state_next,
              mean_state_filt,
              var_state_filt,
              mean_state_pred,
              var_state_pred,
              wgt_state,
              var_state,
              *args, **kwargs):
    r"""
    Performs one step of the square-root Kalman mean/variance smoother.

    Calculates the mean and square-root variance of :math:`p(X_n | Z_{0:N})`.

    Args:
        mean_state_next (ndarray(n_state)): State mean at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, N`.
        var_state_next (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, N`.
        mean_state_filt (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        var_state_filt (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        mean_state_pred (ndarray(n_state)): State mean at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, n`.
        var_state_pred (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, n`.
        wgt_state (ndarray(n_state, n_state)): State transition matrix at time :math:`t = n+1`.
        var_state (ndarray(n_state, n_state)): State square-root variance matrix at time :math:`t = n`.
        args (Optional[pytree]): Additional positional arguments for Kalman functions.
        kwargs (Optional[pytree]): Additional keyword arguments for Kalman functions.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
        - **var_state_smooth** (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
    """

    var_state_temp_tilde = _smooth(
        var_state_filt, var_state_pred, wgt_state
    )
    mean_state_smooth = mean_state_filt + \
        var_state_temp_tilde.dot(mean_state_next - mean_state_pred)
    I = jnp.eye(var_state_temp_tilde.shape[0])    
    J = I - jnp.matmul(var_state_temp_tilde, wgt_state)
    var_state_smooth = add_sqrt(jnp.matmul(var_state_temp_tilde, jnp.hstack((var_state_next, var_state))),
                                jnp.matmul(J,var_state_filt))
    return mean_state_smooth, var_state_smooth


def smooth_sim(x_state_next,
               mean_state_filt,
               var_state_filt,
               mean_state_pred,
               var_state_pred,
               wgt_state,
               var_state,
               *args, **kwargs):
    r"""
    Performs one step of the square-root Kalman sampling smoother.

    Calculates the mean and square-root variance of :math:`p(X_n | X_{n+1}, Z_{0:N})`.

    Args:
        x_state_next (ndarray(n_state)): Simulated state at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, N`.
        mean_state_filt (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        var_state_filt (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        mean_state_pred (ndarray(n_state)): State mean at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, n`.
        var_state_pred (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, n`.
        wgt_state (ndarray(n_state, n_state)): State transition matrix at time :math:`t = n+1`.
        var_state (ndarray(n_state, n_state)): State square-root variance matrix at time :math:`t = n`.
        args (Optional[pytree]): Additional positional arguments for Kalman functions.
        kwargs (Optional[pytree]): Additional keyword arguments for Kalman functions.

    Returns:
        (tuple):
        - **mean_state_sim** (ndarray(n_state)): Mean for the sample solution at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
        - **var_state_sim** (ndarray(n_state)): Square-root variance for the sample solution at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
    """

    var_state_temp_tilde = _smooth(
        var_state_filt, var_state_pred, wgt_state
    )
    mean_state_sim = mean_state_filt + \
        var_state_temp_tilde.dot(x_state_next - mean_state_pred)
    I = jnp.eye(var_state_temp_tilde.shape[0])    
    J = I - jnp.matmul(var_state_temp_tilde, wgt_state)
    var_state_sim = add_sqrt(jnp.matmul(var_state_temp_tilde, var_state),
                             jnp.matmul(J,var_state_filt))
    return mean_state_sim, var_state_sim


def smooth(x_state_next,
           mean_state_next,
           var_state_next,
           mean_state_filt,
           var_state_filt,
           mean_state_pred,
           var_state_pred,
           wgt_state,
           var_state,
           *args, **kwargs):
    r"""
    Performs one step of both square-root Kalman mean/variance and sampling smoothers.

    Args:
        x_state_next (ndarray(n_state)): Simulated state at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, N`.
        mean_state_next (ndarray(n_state)): State mean at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, N`.
        var_state_next (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, N`.
        mean_state_filt (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        var_state_filt (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        mean_state_pred (ndarray(n_state)): State mean at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, n`.
        var_state_pred (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, n`.
        wgt_state (ndarray(n_state, n_state)): State transition matrix at time :math:`t = n+1`.
        var_state (ndarray(n_state, n_state)): State square-root variance matrix at time :math:`t = n`.
        args (Optional[pytree]): Additional positional arguments for Kalman functions.
        kwargs (Optional[pytree]): Additional keyword arguments for Kalman functions.

    Returns:
        (tuple):
        - **mean_state_sim** (ndarray(n_state)): Mean for the sample solution at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
        - **var_state_sim** (ndarray(n_state)): Square-root variance for the sample solution at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
        - **mean_state_smooth** (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
        - **var_state_smooth** (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
    """

    var_state_temp_tilde = _smooth(
        var_state_filt, var_state_pred, wgt_state
    )
    mean_state_temp = jnp.concatenate([x_state_next[None],
                                       mean_state_next[None]])
    mean_state_temp = mean_state_filt + \
        var_state_temp_tilde.dot((mean_state_temp - mean_state_pred).T).T
    mean_state_sim = mean_state_temp[0]
    mean_state_smooth = mean_state_temp[1]
    I = jnp.eye(var_state_temp_tilde.shape[0])    
    J = I - jnp.matmul(var_state_temp_tilde, wgt_state)
    var_state_sim = add_sqrt(jnp.matmul(var_state_temp_tilde, var_state),
                             jnp.matmul(J,var_state_filt))
    var_state_smooth = add_sqrt(jnp.matmul(var_state_temp_tilde, 
                                           jnp.hstack((var_state_next, var_state))),
                                jnp.matmul(J,var_state_filt))
    return mean_state_sim, var_state_sim, mean_state_smooth, var_state_smooth


def forecast(mean_state_pred,
             var_state_pred,
             mean_meas,
             wgt_meas,
             var_meas,
             *args, **kwargs):
    r"""
    Forecasts the mean and variance of the measurement at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.

    Args:
        mean_state_pred (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
        var_state_pred (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
        mean_meas (ndarray(n_meas)): Measurement equation offset at time :math:`t = n`.
        wgt_meas (ndarray(n_meas, n_state)): Measurement transition matrix at time :math:`t = n`.
        var_meas (ndarray(n_meas, n_meas)): State square-root variance matrix at time :math:`t = n`.
        args (Optional[pytree]): Additional positional arguments for Kalman functions.
        kwargs (Optional[pytree]): Additional keyword arguments for Kalman functions.

    Returns:
        (tuple):
        - **mean_fore** (ndarray(n_meas)): Forecast mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
        - **var_fore** (ndarray(n_meas, n_meas)): Forecast variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n-1`.
    """

    # wgt_meas = W + wgt_meas
    mean_fore = wgt_meas.dot(mean_state_pred) + mean_meas
    var_fore = add_sqrt(wgt_meas.dot(var_state_pred), var_meas)   
    var_fore = var_fore.dot(var_fore.T)
    return mean_fore, var_fore


def smooth_cond(mean_state_filt,
                var_state_filt,
                mean_state_pred,
                var_state_pred,
                wgt_state,
                var_state,
                *args, **kwargs):
    r"""
    Performs one step of the square-root Kalman sampling smoother conditional.

    Finds :math:`A_n`, :math:`b_n`, and :math:`C_n = \text{cholesky}(V_n)` such that :math:`X_n \sim \mathcal{N}(A_n X_{n+1} + b_n, V_n)`. This is similar to the Kalman sampling algorithm but without the explicit sample.

    Args:
        mean_state_filt (ndarray(n_state)): State mean at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        var_state_filt (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n` given observations at times :math:`t = 0, \dots, n`.
        mean_state_pred (ndarray(n_state)): State mean at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, n`.
        var_state_pred (ndarray(n_state, n_state)): State square-root variance at time :math:`t = n+1` given observations at times :math:`t = 0, \dots, n`.
        wgt_state (ndarray(n_state, n_state)): State transition matrix at time :math:`t = n+1`.
        var_state (ndarray(n_state, n_state)): State square-root variance matrix at time :math:`t = n`.
        args (Optional[pytree]): Additional positional arguments for Kalman functions.
        kwargs (Optional[pytree]): Additional keyword arguments for Kalman functions.

    Returns:
        (tuple):
        - **wgt_state_cond** (ndarray(n_state, n_state)): Transition of smooth conditional at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
        - **mean_state_cond** (ndarray(n_state)): Offset of smooth conditional at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
        - **var_state_cond** (ndarray(n_state, n_state)): Square-root variance of smooth conditional at time :math:`t = n` given observations at times :math:`t = 0, \dots, N`.
    """

    wgt_state_cond = _smooth(
        var_state_filt, var_state_pred, wgt_state
    )
    mean_state_cond = mean_state_filt - wgt_state_cond.dot(mean_state_pred)
    I = jnp.eye(wgt_state_cond.shape[0])    
    J = I - jnp.matmul(wgt_state_cond, wgt_state)
    var_state_cond = add_sqrt(jnp.matmul(wgt_state_cond, var_state),
                             jnp.matmul(J,var_state_filt))
    return wgt_state_cond, mean_state_cond, var_state_cond


