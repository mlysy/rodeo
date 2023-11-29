r"""
We implement some ready to use interrogation methods from our paper. We implement the interrogation method of Chkrebtii et al (2016), Schober et al (2019) and Kramer et al (2021). 

We also implement two other interrogation methods corresponding to Schober et al (2019) and Kramer et al (2021) where we instead use the variance of Chkrebtii et al (2016). 

"""


import jax
import jax.numpy as jnp


def interrogate_chkrebtii(key, ode_fun, ode_weight, t,
                          mean_state_pred, var_state_pred,
                          **params):
    r"""
    Interrogate method of Chkrebtii et al (2016); DOI: 10.1214/16-BA1017.

    Same arguments and returns as :func:`~ode.interrogate_rodeo`.

    """
    n_block, n_bstate = mean_state_pred.shape
    key, *subkeys = jax.random.split(key, num=n_block+1)
    subkeys = jnp.array(subkeys)
    var_meas = jax.vmap(lambda wm, vsp:
                        jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
        ode_weight, var_state_pred
    )
    x_state = jax.vmap(lambda b:
                       jax.random.multivariate_normal(
                           subkeys[b],
                           mean_state_pred[b],
                           var_state_pred[b]
                       ))(jnp.arange(n_block))
    mean_meas = -ode_fun(x_state, t, **params)
    return jnp.zeros(ode_weight.shape), mean_meas, var_meas


def interrogate_schober(key, ode_fun, ode_weight, t,
                        mean_state_pred, var_state_pred,
                        **params):
    r"""
    Interrogate method of Schober et al (2019); DOI: https://doi.org/10.1007/s11222-017-9798-7.

    Same arguments and returns as :func:`~ode.interrogate_rodeo`.

    """
    n_block, n_bmeas, _ = ode_weight.shape
    var_meas = jnp.zeros((n_block, n_bmeas, n_bmeas))
    mean_meas = -ode_fun(mean_state_pred, t, **params)
    return jnp.zeros(ode_weight.shape), mean_meas, var_meas


def interrogate_kramer(key, ode_fun, ode_weight, t,
                       mean_state_pred, var_state_pred,
                       **params):
    r"""
    First order interrogate method of Kramer et al (2021); DOI: https://doi.org/10.48550/arXiv.2110.11812.
    Assumes off block diagonals are zero.
    Same arguments and returns as :func:`~ode.interrogate_rodeo`.

    """
    n_block, n_bmeas, n_bstate = ode_weight.shape
    fun_meas = -ode_fun(mean_state_pred, t, **params)
    jac = jax.jacfwd(ode_fun)(mean_state_pred, t, **params)
    # need to get the diagonal of jac
    jac = jax.vmap(lambda b:
                   jac[b, :, b])(jnp.arange(n_block))
    wgt_meas = -jac
    mean_meas = jax.vmap(lambda b:
                         fun_meas[b] + jac[b].dot(mean_state_pred[b]))(jnp.arange(n_block))
    var_meas = jnp.zeros((n_block, n_bmeas, n_bmeas))
    return wgt_meas, mean_meas, var_meas


def interrogate_rodeo(key, ode_fun, ode_weight, t,
                      mean_state_pred, var_state_pred,
                      **params):
    r"""
    Rodeo interrogation method.

    Args:
        key (PRNGKey): Jax PRNG key.
        ode_fun (function): Higher order ODE function :math:`W X_t = f(X_t, t, \theta)` taking arguments :math:`X` and :math:`t`.
        ode_weight (ndarray(n_block, n_bmeas, n_bstate)): Weight matrix.
        t (float): Time point.
        mean_state_pred (ndarray(n_block, n_bstate)): Mean estimate for state at time t given observations from times [a...t-1]; denoted by :math:`\mu_{t|t-1}`.
        var_state_pred (ndarray(n_block, n_bstate, n_bstate)): Covariance of estimate for state at time t given observations from times [a...t-1]; denoted by :math:`\Sigma_{t|t-1}`.
        params : Optional model parameters.

    Returns:
        (tuple):
        - **wgt_meas** (ndarray(n_block, n_bmeas, n_bstate)): Interrogation weight matrix.
        - **mean_meas** (ndarray(n_block, n_bmeas)): Interrogation offset.
        - **var_meas** (ndarray(n_block, n_bmeas, n_bmeas)): Interrogation variance.

    """
    n_block = mean_state_pred.shape[0]
    var_meas = jax.vmap(lambda wm, vsp:
                        jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
        ode_weight, var_state_pred
    )
    mean_meas = -ode_fun(mean_state_pred, t, **params)
    return jnp.zeros(ode_weight.shape), mean_meas, var_meas


def interrogate_rodeo2(key, ode_fun, ode_weight, t,
                       mean_state_pred, var_state_pred,
                       **params):
    r"""
    First order interrogate method of Kramer et al (2021); DOI: https://doi.org/10.48550/arXiv.2110.11812.
    Assumes off block diagonals are zero.

    Same arguments and returns as :func:`~ode.interrogate_rodeo`.

    """
    n_block, n_bmeas, n_bstate = ode_weight.shape
    fun_meas = -ode_fun(mean_state_pred, t, **params)
    jac = jax.jacfwd(ode_fun)(mean_state_pred, t, **params)
    # need to get the diagonal of jac
    jac = jax.vmap(lambda b:
                   jac[b, :, b])(jnp.arange(n_block))
    wgt_meas = -jac
    mean_meas = jax.vmap(lambda b:
                         fun_meas[b] + jac[b].dot(mean_state_pred[b]))(jnp.arange(n_block))
    var_meas = jax.vmap(lambda wm, vsp:
                        jnp.atleast_2d(jnp.linalg.multi_dot([wm, vsp, wm.T])))(
        wgt_meas, var_state_pred
    )
    return wgt_meas, mean_meas, var_meas
