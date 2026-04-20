import jax
import jax.numpy as jnp
import tests.interactive.solve as solve
import tests.interactive.utils as utils
from rodeo.utils import multivariate_normal_logpdf


def _update_and_logdens_opt_meas(state_pred, has_meas, x_meas, meas_pars, kalman_funs):
    """
    Perform a joint update and logdensity computation with optional measurement.

    Implemented with `jax.lax.cond()` for use with dynamic `has_meas`.
    """

    def _update_and_logdens_no_meas():
        return state_pred, 0.0

    def _update_and_logdens_meas():
        return utils._update_and_logdens(
            state_pred=state_pred,
            x_meas=x_meas,
            meas_pars=meas_pars,
            kalman_funs=kalman_funs,
        )

    return jax.lax.cond(
        pred=has_meas,
        true_fun=_update_and_logdens_meas,
        false_fun=_update_and_logdens_no_meas,
    )


def fenrir(
    key,
    ode_fun,
    ode_weight,
    ode_init,
    t_min,
    t_max,
    n_steps,
    interrogate,
    prior_pars,
    obs_data,
    obs_times,
    obs_weight,
    obs_var,
    kalman_funs,
):
    """
    Compute marginal logdensity via Fenrir algorithm.

    This also returns the stack of the backward pass for the accompanying ODE.
    May want to make this optional.
    """

    # insert observations on solver time grid
    sim_times = jnp.linspace(t_min, t_max, n_steps + 1)
    obs_ind = jnp.searchsorted(sim_times, obs_times)
    # obs_ind = jnp.full_like(sim_times, fill_value=-1.0).at[obs_ind].set(obs_ind)
    has_meas = jnp.full_like(sim_times, fill_value=False, dtype=bool)
    has_meas = has_meas.at[obs_ind].set(True)

    # full prior
    prior_mean = jnp.zeros_like(ode_init)
    prior_weight, prior_var = prior_pars
    # prior_pars = (prior_weight, prior_mean, prior_var)

    # offset of obs_data assumed to be zero
    obs_mean = jnp.zeros_like(obs_data[0])

    # forward pass
    filter_out = solve.solve_filter(
        key=key,
        ode_fun=ode_fun,
        ode_weight=ode_weight,
        ode_init=ode_init,
        t_min=t_min,
        t_max=t_max,
        n_steps=n_steps,
        interrogate=interrogate,
        prior_pars=prior_pars,
        kalman_funs=kalman_funs,
    )

    # backward pass: scan
    def scan_fun(carry, x):
        # unpack scan variables
        state_past = carry["state_past"]
        logdens = carry["logdens"]
        i_obs = carry["i_obs"]
        state_filt = x["state_filt"]
        state_pred = x["state_pred"]
        has_meas = x["has_meas"]
        # state parameters for backward pass
        bwd_state_pars = kalman_funs.smooth_cond(
            mean_state_filt=state_filt[0],
            var_state_filt=state_filt[1],
            mean_state_pred=state_pred[0],
            var_state_pred=state_pred[1],
            wgt_state=prior_weight,
            var_state=prior_var,
        )
        # backward predict
        bwd_state_pred = kalman_funs.predict(
            mean_state_past=state_past[0],
            var_state_past=state_past[1],
            wgt_state=bwd_state_pars[0],
            mean_state=bwd_state_pars[1],
            var_state=bwd_state_pars[2],
        )

        # backward update
        bwd_state_filt, _logdens = _update_and_logdens_opt_meas(
            state_pred=bwd_state_pred,
            has_meas=has_meas,
            x_meas=obs_data[i_obs],
            meas_pars=(obs_weight[i_obs], obs_mean, obs_var[i_obs]),
            kalman_funs=kalman_funs,
        )

        # output
        carry = {
            "state_past": bwd_state_filt,
            "logdens": logdens + _logdens,
            "i_obs": i_obs - has_meas,
        }
        stack = {
            "state_pred": bwd_state_pred,
            "state_filt": bwd_state_filt,
            "state_pars": bwd_state_pars,
        }
        return carry, stack

    # backward pass: init
    last_state_pred = jax.tree.map(lambda x: x[-1], filter_out["state_filt"])
    i_obs = obs_data.shape[0] - 1
    last_state_filt, last_logdens = _update_and_logdens_opt_meas(
        state_pred=last_state_pred,
        has_meas=has_meas[-1],
        x_meas=obs_data[i_obs],
        meas_pars=(obs_weight[i_obs], obs_mean, obs_var[i_obs]),
        kalman_funs=kalman_funs,
    )
    i_obs = i_obs - has_meas[-1]
    scan_init = {"state_past": last_state_filt, "logdens": last_logdens, "i_obs": i_obs}

    # backward pass: xs
    scan_xs = {
        "state_filt": jax.tree.map(lambda x: x[:-1], filter_out["state_filt"]),
        "state_pred": jax.tree.map(lambda x: x[1:], filter_out["state_pred"]),
        "has_meas": has_meas[:-1],
    }

    # backward pass
    carry, stack = jax.lax.scan(
        f=scan_fun,
        init=scan_init,
        xs=scan_xs,
        reverse=True,
    )

    # append initial values
    stack["state_pred"] = utils.append_last(stack["state_pred"], last_state_pred)
    stack["state_filt"] = utils.append_last(stack["state_filt"], last_state_filt)

    return carry["logdens"], stack
