import jax
import jax.numpy as jnp
import tests.interactive.solve as solve
import tests.interactive.utils as utils

# from rodeo.utils import multivariate_normal_logpdf


def _combine_meas_obs(x_meas, meas_pars, x_obs, obs_pars):
    """
    Combine two sets of measurements.

    Note: This only works for the block parametrization assumed by Kalman
        "standard" and "square-root" algorithms.
    """
    wgt_meas_obs = jnp.concatenate([meas_pars[0], obs_pars[0]], axis=1)
    mean_meas_obs = jnp.concatenate([meas_pars[1], obs_pars[1]], axis=1)
    x_meas_obs = jnp.concatenate([x_meas, x_obs], axis=1)
    var_meas_obs = jax.vmap(jax.scipy.linalg.block_diag)(meas_pars[2], obs_pars[2])
    return x_meas_obs, (wgt_meas_obs, mean_meas_obs, var_meas_obs)


def _update_and_logdens_opt_obs(
    state_pred,
    has_obs,
    x_obs,
    obs_pars,
    x_meas,
    meas_pars,
    kalman_funs,
):
    """
    Perform a joint update and logdensity computation with optional observation,
    in addition to the zero measurement.

    Implemented with `jax.lax.cond()` for use with dynamic `has_obs`.
    """

    def _update_and_logdens_no_obs():
        return utils._update_and_logdens(
            state_pred=state_pred,
            x_meas=x_meas,
            meas_pars=meas_pars,
            kalman_funs=kalman_funs,
        )

    def _update_and_logdens_obs():
        x_meas_obs, meas_obs_pars = _combine_meas_obs(
            x_meas=x_meas, meas_pars=meas_pars, x_obs=x_obs, obs_pars=obs_pars
        )
        return utils._update_and_logdens(
            state_pred=state_pred,
            x_meas=x_meas_obs,
            meas_pars=meas_obs_pars,
            kalman_funs=kalman_funs,
        )

    return jax.lax.cond(
        pred=has_obs,
        true_fun=_update_and_logdens_obs,
        false_fun=_update_and_logdens_no_obs,
    )


def _update_opt_obs(
    state_pred,
    has_obs,
    x_obs,
    i_obs,
    loglik_fun,
    x_meas,
    meas_pars,
    kalman_funs,
):
    """
    Update with optional generalized observation.
    """

    def _update_no_obs():
        return kalman_funs.update(
            mean_state_pred=state_pred[0],
            var_state_pred=state_pred[1],
            x_meas=x_meas,
            wgt_meas=meas_pars[0],
            mean_meas=meas_pars[1],
            var_meas=meas_pars[2],
        )

    def _update_obs():
        # problem dimensions: n_block, n_meas_per_block = n_deriv
        n_block = x_obs.shape[0]
        # gradient and hessian of loglikelihood component
        loglik_grad = jax.jacrev(loglik_fun, argnums=1)
        loglik_hess = jax.jacfwd(loglik_grad, argnums=1)
        obs_grad = loglik_grad(x_obs, state_pred[0], i_obs)
        obs_hess = loglik_hess(x_obs, state_pred[0], i_obs)
        # compute pseudo observations
        # variance term: (n_block, n_deriv, n_block, n_deriv)
        obs_var = jax.vmap(
            lambda b: -jnp.linalg.pinv(obs_hess[b, :, b, :], hermitian=True)
        )(jnp.arange(n_block))
        obs_var = jnp.abs(obs_var)
        # weight term: (n_block, n_deriv, n_deriv)
        obs_weight = jnp.where(jnp.abs(obs_var) > 0.0, 1.0, 0.0)
        obs_mean = jnp.zeros_like(state_pred[0])
        obs_pars = (obs_weight, obs_mean, obs_var)
        x_pseudo = jax.vmap(
            lambda b: obs_weight[b].dot(state_pred[0][b]) + obs_var[b].dot(obs_grad[b])
        )(jnp.arange(n_block))
        # concatenate observations
        x_meas_obs, meas_obs_pars = _combine_meas_obs(
            x_meas=x_meas, meas_pars=meas_pars, x_obs=x_pseudo, obs_pars=obs_pars
        )
        out = kalman_funs.update(
            mean_state_pred=state_pred[0],
            var_state_pred=state_pred[1],
            x_meas=x_meas_obs,
            wgt_meas=meas_obs_pars[0],
            mean_meas=meas_obs_pars[1],
            var_meas=meas_obs_pars[2],
        )
        return out

    return jax.lax.cond(
        pred=has_obs,
        true_fun=_update_obs,
        false_fun=_update_no_obs,
    )


def _smooth_sim_and_logdens(
    x_state_curr,
    x_state_next,
    state_filt,
    state_pred,
    state_pars,
    kalman_funs,
):
    """
    Update the smoothing simulation parameters and evaluate the corresponding
    log-density.
    """
    state_smooth = kalman_funs.smooth_sim(
        x_state_next=x_state_next,
        mean_state_filt=state_filt[0],
        var_state_filt=state_filt[1],
        mean_state_pred=state_pred[0],
        var_state_pred=state_pred[1],
        wgt_state=state_pars[0],
        var_state=state_pars[2],
    )
    lp = kalman_funs.multivariate_normal_logpdf(
        x=x_state_curr, mean=state_smooth[0], cov=state_smooth[1]
    )
    return state_smooth, lp


def dalton(
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
    Compute marginal logdensity via Dalton algorithm.

    This version does not return the stack of the filter, which could be useful
    for generating the corresponding draw of the ODE.
    May want to eventually make that optional.
    """
    # problem dimensions
    n_block, n_eq_per_block, n_deriv = ode_weight.shape
    n_obs, n_block, n_meas_per_block = obs_data.shape

    # solver and observation times
    sim_times, obs_ind, has_obs = utils._process_times(obs_times, t_min, t_max, n_steps)

    # initial values
    mean_state_init = ode_init
    var_state_init = jnp.zeros((n_block, n_deriv, n_deriv))
    state_init = (mean_state_init, var_state_init)

    # full prior
    prior_mean = jnp.zeros_like(ode_init)
    prior_weight, prior_var = prior_pars
    prior_pars = (prior_weight, prior_mean, prior_var)

    # offset of obs_data assumed to be zero
    obs_mean = jnp.zeros_like(obs_data[0])

    # zero measurement variable
    zero_meas = jnp.zeros((n_block, n_eq_per_block))

    # forward pass: scan
    def scan_fun(carry, x):
        # unpack scan variables
        state_past_z, state_past_zy = carry["state_past"]
        logdens_z, logdens_zy = carry["logdens"]
        i_obs = carry["i_obs"]
        t = x["t"]
        key = x["key"]
        has_obs = x["has_obs"]

        # z and y step
        # predict and interrogate
        state_pred_zy, meas_pars_zy = utils._predict_and_interrogate(
            state_past=state_past_zy,
            prior_pars=prior_pars,
            key=key,
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=t,
            interrogate=interrogate,
            kalman_funs=kalman_funs,
        )
        # update and logdens
        state_filt_zy, _logdens_zy = _update_and_logdens_opt_obs(
            state_pred=state_pred_zy,
            has_obs=has_obs,
            x_obs=obs_data[i_obs],
            obs_pars=(obs_weight[i_obs], obs_mean, obs_var[i_obs]),
            x_meas=zero_meas,
            meas_pars=meas_pars_zy,
            kalman_funs=kalman_funs,
        )

        # z only step
        # predict and interrogate
        state_pred_z, meas_pars_z = utils._predict_and_interrogate(
            state_past=state_past_z,
            prior_pars=prior_pars,
            key=key,
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=t,
            interrogate=interrogate,
            kalman_funs=kalman_funs,
        )
        # update and logdens
        state_filt_z, _logdens_z = utils._update_and_logdens(
            state_pred=state_pred_z,
            x_meas=zero_meas,
            meas_pars=meas_pars_z,
            kalman_funs=kalman_funs,
        )

        # output
        carry = {
            "state_past": (state_filt_z, state_filt_zy),
            "logdens": (logdens_z + _logdens_z, logdens_zy + _logdens_zy),
            "i_obs": i_obs + has_obs,
        }
        stack = {
            "state_pred": (state_pred_z, state_pred_zy),
            "state_filt": (state_filt_z, state_filt_zy),
        }
        return carry, stack

    # forward pass: init
    # computation of logdens_zy is somewhat roundabout but easier to generalize
    # to different kalman_funs
    i_obs = 0
    _, logdens_zy = _update_and_logdens_opt_obs(
        state_pred=state_init,
        has_obs=has_obs[0],
        x_obs=obs_data[i_obs],
        obs_pars=(obs_weight[i_obs], obs_mean, obs_var[i_obs]),
        x_meas=zero_meas,
        # meas_pars: all zeros at t=0
        meas_pars=(
            jnp.zeros_like(ode_weight),
            zero_meas,
            jnp.zeros((n_block, n_eq_per_block, n_eq_per_block)),
        ),
        kalman_funs=kalman_funs,
    )
    scan_init = {
        "state_past": (state_init, state_init),
        "logdens": (0.0, logdens_zy),
        "i_obs": i_obs + has_obs[0],
    }

    # forward pass: xs
    scan_xs = {
        "t": sim_times[1:],
        "has_obs": has_obs[1:],
    }
    if key is not None:
        scan_xs["key"] = jax.random.split(key, num=n_steps)
    else:
        scan_xs["key"] = jnp.zeros(n_steps)

    carry, stack = jax.lax.scan(
        f=scan_fun,
        init=scan_init,
        xs=scan_xs,
    )

    logdens_z, logdens_zy = carry["logdens"]
    return logdens_zy - logdens_z


def daltonng(
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
    obs_loglik_i,
    kalman_funs,
):
    """
    Compute marginal logdensity via non-Gaussian Dalton algorithm.

    TODO: The backward pass should contain the output of smooth_cond, i.e.,
        what is needed to simulate a draw from p(X | Z, Y).  This would also
        require computing linear combinations of the form A X + b for the
        specific kalman_funs...
    """
    # problem dimensions
    n_block, n_eq_per_block, n_deriv = ode_weight.shape
    n_obs, n_block, n_meas_per_block = obs_data.shape

    # solver and observation times
    sim_times, obs_ind, has_obs = utils._process_times(obs_times, t_min, t_max, n_steps)

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

    # forward pass: scan
    def scan_fun(carry, x):
        # unpack scan variables
        state_past_z, state_past_zy = carry["state_past"]
        i_obs = carry["i_obs"]
        t = x["t"]
        key = x["key"]
        has_obs = x["has_obs"]

        # predict and interrogate
        state_pred_zy, meas_pars_zy = utils._predict_and_interrogate(
            state_past=state_past_zy,
            prior_pars=prior_pars,
            key=key,
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=t,
            interrogate=interrogate,
            kalman_funs=kalman_funs,
        )
        # update
        state_filt_zy = _update_opt_obs(
            state_pred=state_pred_zy,
            has_obs=has_obs,
            x_obs=obs_data[i_obs],
            i_obs=i_obs,
            loglik_fun=obs_loglik_i,
            x_meas=zero_meas,
            meas_pars=meas_pars_zy,
            kalman_funs=kalman_funs,
        )

        # z only step
        # predict and interrogate
        state_pred_z, meas_pars_z = utils._predict_and_interrogate(
            state_past=state_past_z,
            prior_pars=prior_pars,
            key=key,
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            t=t,
            interrogate=interrogate,
            kalman_funs=kalman_funs,
        )
        # update
        state_filt_z = kalman_funs.update(
            mean_state_pred=state_pred_z[0],
            var_state_pred=state_pred_z[1],
            x_meas=zero_meas,
            wgt_meas=meas_pars_z[0],
            mean_meas=meas_pars_z[1],
            var_meas=meas_pars_z[2],
        )

        # output
        carry = {
            "state_past": (state_filt_z, state_filt_zy),
            "i_obs": i_obs + has_obs,
        }
        stack = {
            "state_pred": (state_pred_z, state_pred_zy),
            "state_filt": (state_filt_z, state_filt_zy),
        }
        return carry, stack

    # forward pass: init
    # update_init is trivial since initial state is known
    scan_init = {"state_past": (state_init, state_init)}
    scan_init["i_obs"] = 0 + has_obs[0]

    # forward pass: xs
    scan_xs = {
        "t": sim_times[1:],
        "has_obs": has_obs[1:],
    }
    if key is not None:
        scan_xs["key"] = jax.random.split(key, num=n_steps)
    else:
        scan_xs["key"] = jnp.zeros(n_steps)

    scan_fun(scan_init, jax.tree.map(lambda x: x[0], scan_xs))

    # forward pass
    fwd_carry, fwd_stack = jax.lax.scan(
        f=scan_fun,
        init=scan_init,
        xs=scan_xs,
    )
    # append initial values
    fwd_stack = utils.append_first(
        fwd_stack,
        first={
            "state_pred": (state_init, state_init),
            "state_filt": (state_init, state_init),
        },
    )
    return fwd_stack

    # backward pass: scan
    def scan_fun(carry, x):
        # unpack scan variables
        x_state_next = carry["x_state_next"]
        state_next = carry["state_next"]
        logdens_z, logdens_zy = carry["logdens"]
        state_filt_z, state_filt_zy = x["state_filt"]
        state_pred_z, state_pred_zy = x["state_pred"]
        # compute x_state_curr
        state_curr = kalman_funs.smooth_mv(
            mean_state_next=state_next[0],
            var_state_next=state_next[1],
            mean_state_filt=state_filt_zy[0],
            var_state_filt=state_filt_zy[1],
            mean_state_pred=state_pred_zy[0],
            var_state_pred=state_pred_zy[1],
            wgt_state=prior_weight,
            var_state=prior_var,
        )
        x_state_curr = state_curr[0]
        # z and y step
        _, _logdens_zy = _smooth_sim_and_logdens(
            x_state_curr=x_state_curr,
            x_state_next=x_state_next,
            state_filt=state_filt_zy,
            state_pred=state_pred_zy,
            state_pars=prior_pars,
            kalman_funs=kalman_funs,
        )
        # z only step
        _, _logdens_z = _smooth_sim_and_logdens(
            x_state_curr=x_state_curr,
            x_state_next=x_state_next,
            state_filt=state_filt_z,
            state_pred=state_pred_z,
            state_pars=prior_pars,
            kalman_funs=kalman_funs,
        )

        # output
        carry = {
            "x_state_next": x_state_curr,
            "state_next": state_curr,
            "logdens": (logdens_z + _logdens_z, logdens_zy + _logdens_zy),
        }
        # for now, just the value used to evaluate p(X | Z), p(X | Z, Y), and p(Y | X)
        stack = x_state_curr

        return carry, stack

    # backward pass: init
    x_state_last = fwd_stack["state_filt"][1][0][-1]
    state_last_zy = (
        fwd_stack["state_filt"][1][0][-1],
        fwd_stack["state_filt"][1][1][-1],
    )
    state_last_z = (
        fwd_stack["state_filt"][0][0][-1],
        fwd_stack["state_filt"][0][1][-1],
    )
    logdens_zy = kalman_funs.multivariate_normal_logpdf(
        x=x_state_last,
        mean=state_last_zy[0],
        cov=state_last_zy[1],
    )
    logdens_z = kalman_funs.multivariate_normal_logpdf(
        x=x_state_last,
        mean=state_last_z[0],
        cov=state_last_z[1],
    )
    scan_init = {
        "x_state_next": x_state_last,
        "state_next": state_last_zy,
        "logdens": (logdens_z, logdens_zy),
    }

    # backward pass: xs
    scan_xs = {
        "state_filt": jax.tree.map(lambda x: x[:-1], fwd_stack["state_filt"]),
        "state_pred": jax.tree.map(lambda x: x[1:], fwd_stack["state_pred"]),
    }

    # backward pass
    bwd_carry, bwd_stack = jax.lax.scan(
        f=scan_fun,
        init=scan_init,
        xs=scan_xs,
        reverse=True,
    )
    logdens_z, logdens_zy = bwd_carry["logdens"]
    x_state_eval = utils.append_last(bwd_stack, last=x_state_last)

    # compute log p(Y | X)
    logdens_x = jax.vmap(obs_loglik_i)(
        obs_data, x_state_eval[has_obs], jnp.arange(n_obs)
    )
    logdens_x = jnp.sum(logdens_x)

    return logdens_x, logdens_z, logdens_zy
