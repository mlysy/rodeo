r"""
For-loop version of the stochastic block solver for ODE initial value problems.

Stochastic block solver for ODE initial value problems.

The ODE-IVP to be solved is defined as

.. math:: W X_t = f(X_t, t, \theta)

on the time interval :math:`t \in [a, b]` with initial condition :math:`X_a = x_0`.  

The stochastic solver proceeds via Kalman filtering and smoothing of "interrogations" of the ODE model as described in Chkrebtii et al 2016, Schober et al 2019.  In the context of the underlying Kalman filterer/smoother, the Gaussian state-space model is

.. math::

    X_n = c_n + Q_n x_{n-1} + R_n^{1/2} \epsilon_n

    z_n = W_n X_n - f(X_n, t, \theta) + V_n^{1/2} \eta_n.

We assume that :math:`c_n = c, Q_n = Q, R_n = R`, and :math:`W_n = W` for all :math:`n`.

This module optimizes the calculations when :math:`Q`, :math:`R`, and :math:`W`, are block diagonal matrices of conformable and "stackable" sizes.  That is, recall that the dimension of these matrices are `n_state x n_state`, `n_state x n_state`, and `n_meas x n_state`, respectively.  Then suppose that :math:`Q` and :math:`R` consist of `n_block` blocks of size `n_bstate x n_bstate`, where `n_bstate = n_state/n_block`, and :math:`W` consists of `n_block` blocks of size `n_bmeas x n_bstate`, where `n_bmeas = n_meas/n_block`.  Then :math:`Q`, :math:`R`, :math:`W` can be stored as 3D arrays of size `n_block x n_bstate x n_bstate` and `n_block x n_bmeas x n_bstate`.  It is under this paradigm that the `ode` module operates.

"""

import jax
import jax.numpy as jnp
from rodeo.kalmantv import *
from rodeo.utils import *


def interrogate_rodeo(key, ode_fun, ode_weight, t,
                      mean_state_pred, var_state_pred,
                      **params):

    n_block, n_bmeas, _ = ode_weight.shape
    var_meas = jnp.zeros((n_block, n_bmeas, n_bmeas))
    for i in range(n_block):
        var_meas = var_meas.at[i].set(jnp.linalg.multi_dot([ode_weight[i], var_state_pred[i], ode_weight[i].T]))

    x_meas = -ode_fun(mean_state_pred, t, **params)
    # var_meas = jnp.array(var_meas)
    return jnp.zeros(ode_weight.shape), x_meas, var_meas


def interrogate_chkrebtii(key, ode_fun, ode_weight, t,
                          mean_state_pred, var_state_pred,
                          **params):
 
    n_block, n_bmeas, n_bstate = ode_weight.shape
    key, *subkeys = jax.random.split(key, num=n_block+1)
    var_meas = jnp.zeros((n_block, n_bmeas, n_bmeas))
    x_state = jnp.zeros((n_block, n_bstate))
    for i in range(n_block):
        var_meas = var_meas.at[i].set(jnp.linalg.multi_dot([ode_weight[i], var_state_pred[i], ode_weight[i].T]))
        x_state = x_state.at[i].set(jax.random.multivariate_normal(subkeys[i], mean_state_pred[i], var_state_pred[i]))
    x_meas = -ode_fun(x_state, t, **params)
    # var_meas = jnp.array(var_meas)
    return jnp.zeros(ode_weight.shape), x_meas, var_meas

def interrogate_kramer(key, ode_fun, ode_weight, t,
                       mean_state_pred, var_state_pred,
                       **params):

    n_block, n_bmeas, n_bstate = ode_weight.shape
    fun_meas = -ode_fun(mean_state_pred, t, **params)
    jacf = jax.jacfwd(ode_fun)(mean_state_pred, t, **params)
    jac = jnp.zeros((n_block, n_bmeas, n_bstate))
    mean_meas = jnp.zeros((n_block, n_bmeas))
    for i in range(n_block):
        jac = jac.at[i].set(jacf[i, :, i])
        mean_meas = mean_meas.at[i].set(fun_meas[i] + jac[i].dot(mean_state_pred[i]))
    wgt_meas = -jac
    # var_meas = jax.vmap(lambda wm, vsp:
    #                     jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
    #     wgt_meas, var_state_pred
    # )
    var_meas = jnp.zeros((n_block, n_bmeas, n_bmeas))
    return wgt_meas, mean_meas, var_meas

def _solve_filter(key, ode_fun,  ode_weight, ode_init,
                  t_min, t_max, n_steps,
                  interrogate,
                  prior_weight, prior_var,
                  **params):

    # Dimensions of block, state and measure variables
    n_block, n_bmeas, n_bstate = ode_weight.shape

    # arguments for kalman_filter and kalman_smooth
    mean_meas = jnp.zeros((n_block, n_bmeas))
    mean_state = jnp.zeros((n_block, n_bstate))
    mean_state_filt = jnp.zeros((n_steps+1, n_block, n_bstate))
    mean_state_pred = jnp.zeros((n_steps+1, n_block, n_bstate))
    var_state_filt = jnp.zeros((n_steps+1, n_block, n_bstate, n_bstate))
    var_state_pred = jnp.zeros((n_steps+1, n_block, n_bstate, n_bstate))

    # initialize
    mean_state_filt = mean_state_filt.at[0].set(ode_init)
    mean_state_pred = mean_state_pred.at[0].set(ode_init)
    x_meas = jnp.zeros((n_block, n_bmeas))

    for t in range(n_steps):
        key, subkey = jax.random.split(key)
        for b in range(n_block):
            mean_state_temp, var_state_temp = \
                predict(
                    mean_state_past=mean_state_filt[t, b],
                    var_state_past=var_state_filt[t, b],
                    mean_state=mean_state[b],
                    wgt_state=prior_weight[b],
                    var_state=prior_var[b]
                )
            mean_state_pred = mean_state_pred.at[t+1, b].set(mean_state_temp)
            var_state_pred = var_state_pred.at[t+1, b].set(var_state_temp)
        # model interrogation
        wgt_meas, mean_meas, var_meas = interrogate(
            key=subkey,
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=t_min + (t_max-t_min)*(t+1)/n_steps,
            mean_state_pred=mean_state_pred[t+1],
            var_state_pred=var_state_pred[t+1],
            **params
        )
        for b in range(n_block):
            # kalman update
            mean_state_temp, var_state_temp = \
                update(
                    mean_state_pred=mean_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b],
                    x_meas=x_meas[b],
                    mean_meas=mean_meas[b],
                    wgt_meas=wgt_meas[b]+ode_weight[b],
                    var_meas=var_meas[b]
                )
            mean_state_filt = mean_state_filt.at[t+1, b].set(mean_state_temp)
            var_state_filt = var_state_filt.at[t+1, b].set(var_state_temp)
    return mean_state_pred, var_state_pred, mean_state_filt, var_state_filt


def solve_sim(key, ode_fun,  ode_weight, ode_init,
              t_min, t_max, n_steps,
              interrogate,
              prior_weight, prior_var,
              **params):
    
    n_block, n_bstate, _ = prior_weight.shape
    key, *subkeys = jax.random.split(key, num=n_steps*n_block+1)
    subkeys = jnp.reshape(jnp.array(subkeys), newshape=(n_steps, n_block, 2))
    x_state_smooth = jnp.zeros((n_steps+1, n_block, n_bstate))
    x_state_smooth = x_state_smooth.at[0].set(ode_init)

    # forward pass
    mean_state_pred, var_state_pred, mean_state_filt, var_state_filt = \
        _solve_filter(
            key=key,
            ode_fun=ode_fun, ode_weight=ode_weight, ode_init=ode_init,
            t_min=t_min, t_max=t_max, n_steps=n_steps, 
            interrogate=interrogate,
            prior_weight=prior_weight, prior_var=prior_var,
            **params   
    )

    for b in range(n_block):
        x_state_smooth = x_state_smooth.at[n_steps, b].set(
            jax.random.multivariate_normal(
                subkeys[n_steps-1, b],
                mean_state_filt[n_steps, b],
                var_state_filt[n_steps, b],
                method='svd')
        )

    for t in range(n_steps-1, 0, -1):
        for b in range(n_block):
            mean_state_sim, var_state_sim = smooth_sim(
                    x_state_next=x_state_smooth[t+1, b],
                    wgt_state=prior_weight[b],
                    mean_state_filt=mean_state_filt[t, b],
                    var_state_filt=var_state_filt[t, b],
                    mean_state_pred=mean_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b]
                )
            x_state_smooth = x_state_smooth.at[t, b].set(
                jax.random.multivariate_normal(subkeys[t-1, b], mean_state_sim, var_state_sim, method='svd'))
    
    # x_state_smooth = jnp.reshape(x_state_smooth, newshape=(-1, n_block*n_bstate))
    return x_state_smooth

def solve_mv(key, ode_fun,  ode_weight, ode_init,
             t_min, t_max, n_steps,
             interrogate,
             prior_weight, prior_var,
             **params):

    n_block, n_bstate, _ = prior_weight.shape
    mean_state_smooth = jnp.zeros((n_steps+1, n_block, n_bstate))
    mean_state_smooth = mean_state_smooth.at[0].set(ode_init)
    var_state_smooth = jnp.zeros((n_steps+1, n_block, n_bstate, n_bstate))

    # forward pass
    mean_state_pred, var_state_pred, mean_state_filt, var_state_filt = \
        _solve_filter(
            key=key,
            ode_fun=ode_fun, ode_weight=ode_weight, ode_init=ode_init,
            t_min=t_min, t_max=t_max, n_steps=n_steps, 
            interrogate=interrogate,
            prior_weight=prior_weight, prior_var=prior_var,
            **params   
    )
    
    mean_state_smooth = mean_state_smooth.at[-1].set(mean_state_filt[-1])
    var_state_smooth = var_state_smooth.at[-1].set(var_state_filt[-1])
    # backward pass
    for t in range(n_steps-1, 0, -1):
        for b in range(n_block):
            mean_state_temp, var_state_temp = \
                smooth_mv(
                    mean_state_next=mean_state_smooth[t+1, b],
                    var_state_next=var_state_smooth[t+1, b],
                    wgt_state=prior_weight[b],
                    mean_state_filt=mean_state_filt[t, b],
                    var_state_filt=var_state_filt[t, b],
                    mean_state_pred=mean_state_pred[t+1, b],
                    var_state_pred=var_state_pred[t+1, b],
            )
            mean_state_smooth = mean_state_smooth.at[t, b].set(mean_state_temp)
            var_state_smooth = var_state_smooth.at[t, b].set(var_state_temp)

    return mean_state_smooth, var_state_smooth

