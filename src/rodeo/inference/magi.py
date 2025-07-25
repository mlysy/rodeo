import jax
import jax.numpy as jnp
from rodeo.kalmantv import standard
from rodeo.kalmantv import square_root

def magi_logdens(ode_data_subset,
                 ode_expand,
                 n_active,
                 prior_pars,
                 kalman_type,
                 **params):
    """
    Log-density of MAGI approximation.

    Args:
        ode_data_subset (ndarray(n_steps+1, n_block, n_deriv-1)): Array specifying :math:`U_{0:N}`, the subset of the solution process needed to reconstruct the entire solution with `ode_expand()`.
        ode_expand (Callable): Function taking inputs `ode_data_subset` and `**params` and returning the full solution process :math:`X_{0:N}`.
        n_active (int): Number of active derivatives -- i.e., not those zero-padded -- for the solution process.
        prior_pars (tuple): A tuple containing the weight matrix and the variance matrix defining the solution prior; :math:`Q, R`.
        kalman_type (str): Determine which type of Kalman (standard, square-root) to use.
        **params (kwargs): Parameters to pass to `ode_expand`.

    Returns:
        (float): Value of the logdensity `p(ode_data_subset, Z = 0 | params, prior_pars)`.
    """
    # standard or square-root filter
    if kalman_type == "standard":
        kalman_funs = standard
    elif kalman_type == "square-root":
        kalman_funs = square_root
    else:
        raise NotImplementedError
    
    # setup
    n_vars = ode_data_subset.shape[1]
    ode_state = ode_expand(ode_data_subset, **params)
    n_deriv = ode_state.shape[2]
    # construct `*_meas` parameters
    wgt_meas = jnp.eye(n_active, n_deriv)
    wgt_meas = jnp.stack([wgt_meas] * n_vars)
    mean_meas = jnp.zeros((n_active,))
    mean_meas = jnp.stack([mean_meas] * n_vars)
    var_meas = jnp.zeros((n_active, n_active))
    var_meas = jnp.stack([var_meas] * n_vars)
    
    # construct remaining `*_state` parameters
    mean_state = jnp.zeros((n_vars, n_deriv))
    wgt_state, var_state = prior_pars

    # kalman filter
    def filter_scan(carry, x_meas):
        mean_state_past, var_state_past = carry["state"]
        # kalman predict
        mean_state_pred, var_state_pred = jax.vmap(kalman_funs.predict)(
            mean_state_past=mean_state_past,
            var_state_past=var_state_past,
            mean_state=mean_state,
            wgt_state=wgt_state,
            var_state=var_state
        )
        # kalman forecast (for logdens)
        mean_state_fore, var_state_fore = jax.vmap(kalman_funs.forecast)(
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            mean_meas=mean_meas,
            wgt_meas=wgt_meas,
            var_meas=var_meas
        )
        # var_state_fore = jax.vmap(lambda a: a.dot(a.T))(var_state_fore)
        logdens = jax.vmap(jax.scipy.stats.multivariate_normal.logpdf)(
            x=x_meas,
            mean=mean_state_fore,
            cov=var_state_fore
        )
        # kalman update
        mean_state_next, var_state_next = jax.vmap(kalman_funs.update)(
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            x_meas=x_meas,
            mean_meas=mean_meas,
            wgt_meas=wgt_meas,
            var_meas=var_meas
        )
        carry["state"] = (mean_state_next, var_state_next)
        carry["logdens"] = carry["logdens"] + jnp.sum(logdens)
        return carry, None

    filter_init = {
        "state": (ode_state[0], jnp.zeros((n_vars, n_deriv, n_deriv))),
        "logdens": 0.0
    }

    res, _ = jax.lax.scan(
        f=filter_scan,
        init=filter_init,
        xs=ode_state[1:, :, :n_active]
    )

    return res["logdens"]
