r"""
For-loop version of the stochastic block solver for ODE initial value problems.

In the notation defined in the `ode_solve`, recall that the Gaussian state space model underlying the ODE-IVP is

:: math::

X_n = Q(X_{n-1} - \lambda) + \lambda + R^{1/2} \epsilon_n

y_n = W X_n + \Sigma_n \eta_n.

This module optimizes the calculations when :math:`Q`, :math:`R`, and :math:`W`, are block diagonal matrices of conformable and "stackable" sizes.  That is, recall that the dimension of these matrices are `n_state x n_state`, `n_state x n_state`, and `n_meas x n_state`, respectively.  Then suppose that :math:`Q` and :math:`R` consist of `n_block` blocks of size `n_bstate x n_bstate`, where `n_bstate = n_state/n_block`, and :math:`W` consists of `n_block` blocks of size `n_bmeas x n_bstate`, where `n_bmeas = n_meas/n_block`.  Then :math:`Q`, :math:`R`, :math:`W` can be stored as 3D arrays of size `n_block x n_bstate x n_bstate` and `n_block x n_bmeas x n_bstate`.  It is under this paradigm that the `ode_block_solve` module operates.

"""

# import numpy as np
import jax
import jax.numpy as jnp
import numpy as np
from rodeo.kalmantv import *
from rodeo.utils import *


def interrogate_rodeo(key, fun, W, t, theta,
                      mean_state_pred, var_state_pred):
    r"""
    Rodeo interrogation method.

    Args:
        key (PRNGKey): Jax PRNG key.
        fun (function): Higher order ODE function :math:`W X_t = F(X_t, t)` taking arguments :math:`X` and :math:`t`.
        t (float): Time point.
        theta (ndarray(n_theta)): ODE parameter.
        W (ndarray(n_block, n_bmeas, n_bstate)): Transition matrix defining the measure prior.
        mean_state_pred (ndarray(n_block, n_bstate)): Mean estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\mu_{n|n-1}`.
        var_state_pred (ndarray(n_block, n_bstate, n_bstate)): Covariance of estimate for state at time n given observations from times [0...n-1]; denoted by :math:`\Sigma_{n|n-1}`.

    Returns:
        (tuple):
        - **x_meas** (ndarray(n_block, n_bmeas)): Interrogation variable.
        - **var_meas** (ndarray(n_block, n_bmeas, n_bmeas)): Interrogation variance.

    """
    n_block, n_bmeas, _ = W.shape
    var_meas = np.zeros((n_block, n_bmeas, n_bmeas))
    for i in range(n_block):
        var_meas[i] = np.linalg.multi_dot([W[i], var_state_pred[i], W[i].T])

    x_meas = fun(mean_state_pred, t, theta)
    var_meas = jnp.array(var_meas)
    return x_meas, var_meas


def interrogate_chkrebtii(key, fun, W, t, theta,
                          mean_state_pred, var_state_pred):
    r"""
    Interrogate method of Chkrebtii et al (2016); DOI: 10.1214/16-BA1017.

    Same arguments and returns as :func:`~ode_block_solve.interrogate_rodeo`.

    """
    n_block, n_bmeas, n_bstate = W.shape
    key, *subkeys = jax.random.split(key, num=n_block+1)
    #z_state = jax.random.normal(key, (n_block, n_bstate))
    var_meas = np.zeros((n_block, n_bmeas, n_bmeas))
    x_state = np.zeros((n_block, n_bstate))
    for i in range(n_block):
        var_meas[i] = np.linalg.multi_dot([W[i], var_state_pred[i], W[i].T])
        x_state[i] = jax.random.multivariate_normal(subkeys[i], mean_state_pred[i], var_state_pred[i])
    x_meas = fun(x_state, t, theta)
    var_meas = jnp.array(var_meas)
    return x_meas, var_meas

def _solve_filter(key, fun, W, x0, theta,
                  tmin, tmax, n_steps, 
                  trans_state, mean_state, var_state,
                  interrogate):
    r"""
    Forward pass of the ODE solver.

    key (PRNGKey): PRNG key.
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
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_pred** (ndarray(n_steps, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mean_state_filt** (ndarray(n_steps, n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps, n_block, n_bstate, n_bstate)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = W.shape
    n_eval = n_steps - 1
    #n_state = len(mean_state)

    # arguments for kalman_filter and kalman_smooth
    mean_meas = jnp.zeros((n_block, n_bmeas))
    mean_state_filt = np.zeros((n_steps, n_block, n_bstate))
    mean_state_pred = np.zeros((n_steps, n_block, n_bstate))
    var_state_filt = np.zeros((n_steps, n_block, n_bstate, n_bstate))
    var_state_pred = np.zeros((n_steps, n_block, n_bstate, n_bstate))

    # initialize
    mean_state_filt[0] = x0
    mean_state_pred[0] = x0

    for t in range(n_eval):
        key, subkey = jax.random.split(key)
        for b in range(n_block):
            mean_state_pred[t+1, b], var_state_pred[t+1, b] = \
                predict(
                    mean_state_past=mean_state_filt[t, b],
                    var_state_past=var_state_filt[t, b],
                    mean_state=mean_state[b],
                    trans_state=trans_state[b],
                    var_state=var_state[b]
                )
        # model interrogation
        x_meas, var_meas = interrogate(
            key=subkey,
            fun=fun,
            W=W,
            t=tmin + (tmax-tmin)*(t+1)/n_eval,
            theta=theta,
            mean_state_pred=mean_state_pred[t+1],
            var_state_pred=var_state_pred[t+1]
        )
        for b in range(n_block):
            # kalman update
            mean_state_filt[t+1, b], var_state_filt[t+1, b] = \
                update(
                    mean_state_pred=mean_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b],
                    x_meas=x_meas[b],
                    mean_meas=mean_meas[b],
                    trans_meas=W[b],
                    var_meas=var_meas[b]
                )
    return mean_state_pred, var_state_pred, mean_state_filt, var_state_filt


def solve_sim(key, fun, W, x0, theta,
              tmin, tmax, n_steps,
              trans_state, mean_state, var_state,
              interrogate=interrogate_rodeo):
    r"""
    Random draw from the stochastic ODE solver.

    Args:
        key (PRNGKey): PRNG key.
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
        interrogate (function): Function defining the interrogation method.

    Returns:
        (ndarray(n_steps, n_blocks, n_bstate)): Sample solution for :math:`X_t` at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mean_state.shape
    n_eval = n_steps - 1
    key, *subkeys = jax.random.split(key, num=n_eval*n_block+1)
    subkeys = jnp.reshape(jnp.array(subkeys), newshape=(n_eval, n_block, 2))
    x_state_smooth = np.zeros((n_steps, n_block, n_bstate))
    x_state_smooth[0] = x0

    # forward pass
    mean_state_pred, var_state_pred, mean_state_filt, var_state_filt = \
        _solve_filter(
            key=key,
            fun=fun, W=W, x0=x0, theta=theta,
            tmin=tmin, tmax=tmax, 
            n_steps=n_steps, trans_state=trans_state,
            mean_state=mean_state, var_state=var_state,
            interrogate=interrogate
        )

    # for b in range(n_block):
    #     x_state_smooth[n_eval, b] = \
    #         _state_sim(
    #             mean_state_filt[n_eval, b],
    #             var_state_filt[n_eval, b],
    #             z_state[n_eval-1, b])

    for b in range(n_block):
        x_state_smooth[n_eval, b] = \
            jax.random.multivariate_normal(
                subkeys[n_eval-1, b],
                mean_state_filt[n_eval, b],
                var_state_filt[n_eval, b])

    for t in range(n_eval-1, 0, -1):
        for b in range(n_block):
            mean_state_sim, var_state_sim = smooth_sim(
                    x_state_next=x_state_smooth[t+1, b],
                    trans_state=trans_state[b],
                    mean_state_filt=mean_state_filt[t, b],
                    var_state_filt=var_state_filt[t, b],
                    mean_state_pred=mean_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b]
                )
            x_state_smooth[t, b] = jax.random.multivariate_normal(subkeys[t-1, b], mean_state_sim, var_state_sim)
    
    # x_state_smooth = jnp.reshape(x_state_smooth, newshape=(-1, n_block*n_bstate))
    return jnp.array(x_state_smooth)

def solve_mv(key, fun, W, x0, theta,
             tmin, tmax, n_steps,
             trans_state, mean_state, var_state,
             interrogate=interrogate_rodeo):
    r"""
    Mean and variance of the stochastic ODE solver.

    Args:
        key (PRNGKey): PRNG key.
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
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **mean_state_smooth** (ndarray(n_steps, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mean_state.shape
    n_eval = n_steps - 1
    mean_state_smooth = np.zeros((n_steps, n_block, n_bstate))
    mean_state_smooth[0] = x0
    var_state_smooth = np.zeros((n_steps, n_block, n_bstate, n_bstate))

    # forward pass
    mean_state_pred, var_state_pred, mean_state_filt, var_state_filt = \
        _solve_filter(
            key=key,
            fun=fun, W=W, x0=x0, theta=theta,
            tmin=tmin, tmax=tmax, n_steps=n_steps,
            trans_state=trans_state,
            mean_state=mean_state, var_state=var_state,
            interrogate=interrogate
        )
    
    mean_state_smooth[-1] = mean_state_filt[-1]
    var_state_smooth[-1] = var_state_filt[-1]
    # backward pass
    for t in range(n_eval-1, 0, -1):
        for b in range(n_block):
            mean_state_smooth[t, b], var_state_smooth[t, b] = \
                smooth_mv(
                    mean_state_next=mean_state_smooth[t+1, b],
                    var_state_next=var_state_smooth[t+1, b],
                    trans_state=trans_state[b],
                    mean_state_filt=mean_state_filt[t, b],
                    var_state_filt=var_state_filt[t, b],
                    mean_state_pred=mean_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b],
            )
    # mean_state_smooth = jnp.reshape(mean_state_smooth, newshape=(-1, n_block*n_bstate))
    # var_state_smooth = block_diag(jnp.array(var_state_smooth))
    return jnp.array(mean_state_smooth), jnp.array(var_state_smooth)


def solve(key, fun, W, x0, theta,
          tmin, tmax, n_steps,
          trans_state, mean_state, var_state,
          interrogate=interrogate_rodeo):
    r"""
    Both random draw and mean/variance of the stochastic ODE solver.

    Args:
        key (PRNGKey): PRNG key.
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
        interrogate (function): Function defining the interrogation method.

    Returns:
        (tuple):
        - **x_state_smooth** (ndarray(n_steps, n_block, n_bstate)): Sample solution for :math:`X_t` at times :math:`t \in [a, b]`.
        - **mean_state_smooth** (ndarray(n_steps, n_block, n_bstate)): Posterior mean of the solution process :math:`X_t` at times :math:`t \in [a, b]`.
        - **var_state_smooth** (ndarray(n_steps, n_block, n_bstate, n_bstate)): Posterior variance of the solution process at times :math:`t \in [a, b]`.

    """
    n_block, n_bstate = mean_state.shape
    n_eval = n_steps - 1
    #key, subkey = jax.random.split(key)
    #z_state = jax.random.normal(subkey, (n_eval, n_block, n_bstate))
    key, *subkeys = jax.random.split(key, num=n_eval*n_block+1)
    subkeys = jnp.reshape(jnp.array(subkeys), newshape=(n_eval, n_block, 2))
    x_state_smooth = np.zeros((n_steps, n_block, n_bstate))
    x_state_smooth[0] = x0
    mean_state_smooth = np.zeros((n_steps, n_block, n_bstate))
    mean_state_smooth[0] = x0
    var_state_smooth = np.zeros((n_steps, n_block, n_bstate, n_bstate))

    # forward pass
    mean_state_pred, var_state_pred, mean_state_filt, var_state_filt = \
        _solve_filter(
            key=key,
            fun=fun, W=W, x0=x0, theta=theta,
            tmin=tmin, tmax=tmax, n_steps=n_steps,
            trans_state=trans_state,
            mean_state=mean_state, var_state=var_state,
            interrogate=interrogate
        )

    mean_state_smooth[-1] = mean_state_filt[-1]
    var_state_smooth[-1] = var_state_filt[-1]

    # for b in range(n_block):
    #     x_state_smooth[n_eval, b] = \
    #         _state_sim(
    #             mean_state_filt[n_eval, b],
    #             var_state_filt[n_eval, b],
    #             z_state[n_eval-1, b])

    for b in range(n_block):
        x_state_smooth[n_eval, b] = \
            jax.random.multivariate_normal(
                subkeys[n_eval-1, b],
                mean_state_filt[n_eval, b],
                var_state_filt[n_eval, b])

    # backward pass
    for t in range(n_eval-1, 0, -1):
        for b in range(n_block):
            mean_state_sim, var_state_sim, mean_state_smooth[t, b], var_state_smooth[t, b] = \
                smooth(
                    x_state_next=x_state_smooth[t+1, b],
                    mean_state_next=mean_state_smooth[t+1, b],
                    var_state_next=var_state_smooth[t+1, b],
                    trans_state=trans_state[b],
                    mean_state_filt=mean_state_filt[t, b],
                    var_state_filt=var_state_filt[t, b],
                    mean_state_pred=mean_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b]
                )
            x_state_smooth[t, b] = jax.random.multivariate_normal(subkeys[t-1, b], mean_state_sim, var_state_sim)
    # x_state_smooth = jnp.reshape(x_state_smooth, newshape=(-1, n_block*n_bstate))
    # mean_state_smooth = jnp.reshape(mean_state_smooth, newshape=(-1, n_block*n_bstate))
    # var_state_smooth = block_diag(jnp.array(var_state_smooth))
    return jnp.array(x_state_smooth), jnp.array(mean_state_smooth), jnp.array(var_state_smooth)
