r"""
Time-varying Kalman filtering and smoothing algorithms.

The Gaussian state space model underlying the algorithms is

.. math::

    x_n = \mu(x_{n-1}, \theta) + \Sigma(x_{n-1}, \theta) \epsilon_n

    y_n \sim p(\x_n | \theta),

where :math:`\epsilon_n \stackrel{\text{iid}}{\sim} \operatorname{Normal}(0, I_p)`.  At each time :math:`n`, only :math:`y_n` is observed.  The Kalman filtering and smoothing algorithms efficiently calculate quantities of the form :math:`\theta_{m|n} = (\mu_{m|n}, \Sigma_{m|n})`, where

.. math::

    \mu_{m|n} = E[x_m \mid y_{0:n}]

    \Sigma_{m|n} = \text{var}(x_m \mid y_{0:n}),

for different combinations of :math:`m` and :math:`n`.

"""
import jax
import jax.numpy as jnp
from rodeo import kalmantv
    
def predict(mean_state_past, var_state_past,
            mean_fun, var_fun, theta):
    r"""
    Perform one prediction step of the Extended Kalman filter.

    Calculates :math:`\theta_{n|n-1}` from :math:`\theta_{n-1|n-1}`.

    Args:
        mean_state_past (ndarray(n_state)): Mean estimate for state at time n-1 given observations from times [0...n-1]; :math:`\mu_{n-1|n-1}`.
        var_state_past (ndarray(n_state, n_state)): Covariance of estimate for state at time n-1 given observations from times [0...n-1]; :math:`\Sigma_{n-1|n-1}`.
        mean_fun (func): Function defining the mean of the state.
        var_fun (func): Function defining the variance of the state.
        theta (ndarray(n_theta)): Parameters for the mean and variance functions.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        - **var_state_pred** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        - **wgt_state** (ndarray(n_state, n_state)): Linearized transition matrix for the state.

    """
    mean_state_pred = mean_fun(mean_state_past, theta)
    wgt_state = jax.jacobian(mean_fun)(mean_state_past, theta)
    var_state = var_fun(mean_state_past, theta)
    var_state_pred = jnp.linalg.multi_dot(
        [wgt_state, var_state_past, wgt_state.T]) + var_state
    return mean_state_pred, var_state_pred, wgt_state

def update(mean_state_pred,
           var_state_pred,
           x_meas,
           meas_lpdf,
           theta):
    r"""
    Perform one update step of the Extended Kalman filter.

    Calculates :math:`\theta_{n|n}` from :math:`\theta_{n|n-1}`.

    Args:
        mean_state_pred (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.
        x_meas (ndarray(n_meas)): Measure vector from `x_state`; :math:`y_n`.
        meas_lpdf (func): The logpdf for the measurement vector.
        theta (ndarray(n_theta)): Parameters for the mean and variance functions.

    Returns:
        (tuple):
        - **mean_state_filt** (ndarray(n_state)): Mean estimate for state at time n given observations from times [0...n]; denoted by :math:`\mu_{n|n}`.
        - **var_state_filt** (ndarray(n_state, n_state)): Covariance of estimate for state at time n given observations from times [0...n]; denoted by :math:`\Sigma_{n|n}`.

    """
    # parameters for linear update
    jac = jax.jacobian(meas_lpdf)(mean_state_pred, x_meas, theta)
    hess = jax.hessian(meas_lpdf)(mean_state_pred, x_meas, theta)
    var_meas = -jnp.linalg.inv(hess)
    y_hat = mean_state_pred + var_meas.dot(jac)
    # wgt_meas = jnp.where(var_meas != 0, 1, 0)
    wgt_meas = jnp.eye(len(y_hat))
    
    return kalmantv.update(mean_state_pred,
                           var_state_pred,
                           y_hat,
                           jnp.zeros((len(y_hat,))),
                           wgt_meas,
                           var_meas)