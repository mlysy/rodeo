import jax
import jax.numpy as jnp
import tests.interactive.utils as utils
from rodeo.utils import multivariate_normal_logpdf


def solve_filter(
    key,
    ode_fun,
    ode_weight,
    ode_init,
    t_min,
    t_max,
    n_steps,
    interrogate,
    prior_pars,
    kalman_funs,
):
    # problem dimensions
    n_block, n_eq_per_block, n_deriv = ode_weight.shape

    # initial values
    mean_state_init = ode_init
    var_state_init = jnp.zeros((n_block, n_deriv, n_deriv))
    state_init = (mean_state_init, var_state_init)

    # full prior
    prior_mean = jnp.zeros_like(ode_init)
    prior_weight, prior_var = prior_pars
    prior_pars = (prior_weight, prior_mean, prior_var)

    # zero measurement variable
    zero_meas = jnp.zeros((n_block, n_eq_per_block))

    # scan: xs
    scan_xs = {"t": jnp.linspace(t_min, t_max, num=n_steps + 1)[1:]}
    if key is not None:
        scan_xs["key"] = jax.random.split(key, num=n_steps)
    else:
        scan_xs["key"] = jnp.zeros(n_steps)

    # scan: fun
    def scan_fun(carry, x):
        # unpack scan variables
        state_past = carry
        t = x["t"]
        key = x["key"]
        # predict and interrogate
        state_pred, meas_pars = utils._predict_and_interrogate(
            state_past=state_past,
            prior_pars=prior_pars,
            key=key,
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=t,
            interrogate=interrogate,
            kalman_funs=kalman_funs,
        )
        # breakpoint()
        # print(f"state_pred = {state_pred}, meas_pars = {meas_pars}")
        # update
        state_filt = kalman_funs.update(
            mean_state_pred=state_pred[0],
            var_state_pred=state_pred[1],
            x_meas=zero_meas,
            wgt_meas=meas_pars[0],
            mean_meas=meas_pars[1],
            var_meas=meas_pars[2],
        )
        carry = state_filt
        stack = {"state_filt": state_filt, "state_pred": state_pred}
        return carry, stack

    # scan_fun(state_init, jax.tree.map(lambda x: x[0], scan_xs))
    _, res = jax.lax.scan(f=scan_fun, init=state_init, xs=scan_xs)

    # append initial values
    res = utils.append_first(
        res, first={"state_filt": state_init, "state_pred": state_init}
    )
    return res


def solve_mv(
    key,
    ode_fun,
    ode_weight,
    ode_init,
    t_min,
    t_max,
    n_steps,
    interrogate,
    prior_pars,
    kalman_funs,
):
    filter_out = solve_filter(
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

    # scan functions
    prior_weight, prior_var = prior_pars

    def smooth_fn(state_next, x, kalman_funs):
        return {"wgt_state": prior_weight, "var_state": prior_var}

    return utils.kalman_smooth_mv(
        state_pred=filter_out["state_pred"],
        state_filt=filter_out["state_filt"],
        smooth_fn=smooth_fn,
        kalman_funs=kalman_funs,
    )


def solve_sim(
    key,
    ode_fun,
    ode_weight,
    ode_init,
    t_min,
    t_max,
    n_steps,
    interrogate,
    prior_pars,
    kalman_funs,
):
    subkeys = jax.random.split(key, num=2)

    # forward pass
    filter_out = solve_filter(
        key=subkeys[0],
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

    # scan functions
    prior_weight, prior_var = prior_pars

    def smooth_fn(state_next, x, kalman_funs):
        return {"wgt_state": prior_weight, "var_state": prior_var}

    return utils.kalman_smooth_sim(
        key=subkeys[1],
        state_pred=filter_out["state_pred"],
        state_filt=filter_out["state_filt"],
        smooth_fn=smooth_fn,
        kalman_funs=kalman_funs,
    )
