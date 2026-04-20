import jax
import jax.numpy as jnp

# from rodeo.utils import multivariate_normal_logpdf


def append_last(x, last):
    """
    Append an element to the end of a PyTree.
    """
    return jax.tree.map(lambda _x, _last: jnp.concatenate([_x, _last[None]]), x, last)


def append_first(x, first):
    """
    Append an element to the beginning of a PyTree.
    """
    return jax.tree.map(
        lambda _x, _first: jnp.concatenate([_first[None], _x]), x, first
    )


def multivariate_normal_logpdf(x, mean, cov):
    """
    Log-density of possibly a subset of a multivariate normal.

    Rather than subsetting directly, which may not be compatible with `jax.vmap()`
    and `jax.lax.scan()`, the subset is indicated by the nonzero diagonal elements of `cov`.

    That is, let
    ```
    zero_id = jnp.isclose(jnp.diag(cov), 0.0, rtol=1e-10)
    inf_id = jnp.diag(cov) == jnp.inf
    rm_id = zero_ind | inf_id
    ```
    Then `cov_pd = cov[jnp.idx_(~rm_id, ~rm_id)]` must be positive definite matrix,
    and the remaining off-diagional elements of `cov` must all be zero.
    This function then returns
    ```
    jax.scipy.stats.multivariate_normal.logpdf(x[~rm_id], mean[~rm_id], cov_pd)
    ```

    FIXME: Make this work for both zeros and infs.
    """
    cov_diag = jnp.diag(cov)
    zero_id = jnp.isclose(cov_diag, 0.0, rtol=1e-16)
    inf_id = cov_diag == jnp.inf
    rm_id = zero_id | inf_id
    cov_diag = jnp.where(zero_id, jnp.inf, cov_diag)
    cov = cov.at[jnp.diag_indices(cov_diag.shape[0])].set(cov_diag)
    cfac = jax.scipy.linalg.cholesky(cov)
    z = jax.scipy.linalg.solve_triangular(cfac, x - mean, trans=1)
    cdiag = jnp.diag(cfac)
    cdiag = jnp.where(rm_id, 1.0, cdiag)
    const = jnp.sum(~rm_id) * jnp.log(2.0 * jnp.pi)
    return -(0.5 * (jnp.sum(z**2) + const) + jnp.sum(jnp.log(cdiag)))


def kalman_filter(
    state_init, obs, kalman_funs, predict_fn, update_fn, stack_fn=None, reverse=False
):
    """
    Expanded Kalman filter.

    Args:
        state_init (PyTree): Mean and variance of the state prior to any observations.

        obs (Meas): Generalized observations.  Leading dimension of each leaf is `n_obs`.

        predict_fn (Callable): Function to return the element of the state-space model
            needed to perform the predict step.  Inputs are:
            - `state`: PyTree with same structure as `state_init`.
            - `x`: PyTree consisting of a slice of `obs` across the leading dimension, i.e., as performed by `lax.scan`.
            - `kalman_funs`: See below.
            Output is a dictionary with with names corresponding to the required
            inputs to `kalman_funs.predict()`.

        update_fn (Callable): Funtion to return the elements of the state-space model
            needed to perform the update step.  Inputs are the same as `predict_fn`.
            Return type must be a dictionary with names corresponding to the required
            inputs to `kalman_funs.update()`.

        stack_fn (Callable): Function to create the stack output of `lax.scan()`.  Set to `None` to omit.

        kalman_funs: An object or module from which can be accessed various Kalman
            methods.  The required ones are `kalman_funs.predict()` and
            `kalman_funs.update()`.

        reverse: Whether to run `lax.scan()` in reverse.

    Returns:
        If `stack_fn` is `None`, a dictionary with keys `state_pred` and `state_filt`.
        Each of these are PyTrees the same structure as `state_init` but with added leading dimension `n_obs`.

        If `stack_fn` is not `None`, a tuple where the first element is the above
        and the second is a PyTree having the same structure as the output of
        `stack_fn()`, but with added leading dimension `n_obs`.
    """

    def scan_fun(state_past, x):
        # predict
        state_args = predict_fn(state_past, x, kalman_funs)
        state_pred = kalman_funs.predict(
            mean_state_past=state_past[0],
            var_state_past=state_past[1],
            **state_args,
        )
        # update
        meas_args = update_fn(state_pred, x, kalman_funs)
        state_filt = kalman_funs.update(
            mean_state_pred=state_pred[0],
            var_state_pred=state_pred[1],
            **meas_args,
        )
        # stack
        stack = {"state_pred": state_pred, "state_filt": state_filt}
        if stack_fn is not None:
            stack = (
                stack,
                stack_fn(
                    state_pred=state_pred,
                    state_filt=state_filt,
                    x=x,
                    kalman_funs=kalman_funs,
                    **state_args,
                    **meas_args,
                ),
            )
        # output
        return state_filt, stack

    _, res = jax.lax.scan(
        f=scan_fun,
        init=state_init,
        xs=obs,
        reverse=reverse,
    )
    return res


def kalman_smooth_mv(state_pred, state_filt, smooth_fn, kalman_funs):
    """
    Kalman mean/variance smoother.
    """

    def scan_fn(state_next, x):
        # smooth
        smooth_args = smooth_fn(state_next, x, kalman_funs)
        state_curr = kalman_funs.smooth_mv(
            mean_state_next=state_next[0],
            var_state_next=state_next[1],
            **x,
            **smooth_args,
        )
        return state_curr, state_curr

    state_last = (state_filt[0][-1], state_filt[1][-1])
    smooth_args = {
        "mean_state_filt": state_filt[0][:-1],
        "var_state_filt": state_filt[1][:-1],
        "mean_state_pred": state_pred[0][1:],
        "var_state_pred": state_pred[1][1:],
    }

    _, res = jax.lax.scan(
        f=scan_fn,
        init=state_last,
        xs=smooth_args,
        reverse=True,
    )

    # append initial mv to end
    res = append_last(res, last=state_last)
    # res = (
    #     append_last(res[0], state_last[0]),
    #     append_last(res[1], state_last[1]),
    # )

    return res


def kalman_smooth_sim(key, state_pred, state_filt, smooth_fn, kalman_funs):
    """
    Kalman simulation smoother.
    """

    def scan_fn(x_state_next, x):
        # smooth
        smooth_args = smooth_fn(x_state_next, x, kalman_funs)
        state_curr = kalman_funs.smooth_sim(
            x_state_next=x_state_next,
            **x,
            **smooth_args,
        )
        x_state_curr = jax.random.multivariate_normal(
            key=x["key"], mean=state_curr[0], cov=state_curr[1], method="svd"
        )
        return x_state_curr, x_state_curr

    n_obs = state_filt[0].shape[0]
    subkeys = jax.random.split(key, num=n_obs)
    state_last = (state_filt[0][-1], state_filt[1][-1])
    x_state_last = jax.random.multivariate_normal(
        key=subkeys[-1], mean=state_last[0], cov=state_last[1], method="svd"
    )
    smooth_args = {
        "key": subkeys[:-1],
        "mean_state_filt": state_filt[0][:-1],
        "var_state_filt": state_filt[1][:-1],
        "mean_state_pred": state_pred[0][1:],
        "var_state_pred": state_pred[1][1:],
    }

    _, res = jax.lax.scan(
        f=scan_fn,
        init=x_state_last,
        xs=smooth_args,
        reverse=True,
    )

    # append initial sim to end
    res = append_last(res, x_state_last)

    return res


def _predict_and_interrogate(
    state_past,
    prior_pars,
    key,
    ode_fun,
    ode_weight,
    t,
    interrogate,
    kalman_funs,
):
    """
    Perform a joint predict and interrogate step.
    """
    # predict
    state_pred = kalman_funs.predict(
        mean_state_past=state_past[0],
        var_state_past=state_past[1],
        wgt_state=prior_pars[0],
        mean_state=prior_pars[1],
        var_state=prior_pars[2],
    )
    # interrogate
    wgt_meas, mean_meas, var_meas = interrogate(
        key=key,
        ode_fun=ode_fun,
        ode_weight=ode_weight,
        t=t,
        mean_state_pred=state_pred[0],
        var_state_pred=state_pred[1],
    )
    meas_pars = (ode_weight + wgt_meas, mean_meas, var_meas)
    return state_pred, meas_pars


def _update_and_logdens(state_pred, x_meas, meas_pars, kalman_funs):
    """
    Perform a joint update and logdensity computation.
    """
    # logdens calculation
    state_fore = kalman_funs.forecast(
        mean_state_pred=state_pred[0],
        var_state_pred=state_pred[1],
        wgt_meas=meas_pars[0],
        mean_meas=meas_pars[1],
        var_meas=meas_pars[2],
    )
    logdens = kalman_funs.multivariate_normal_logpdf(
        x=x_meas,
        mean=state_fore[0],
        cov=state_fore[1],
    )
    # update
    state_filt = kalman_funs.update(
        mean_state_pred=state_pred[0],
        var_state_pred=state_pred[1],
        x_meas=x_meas,
        wgt_meas=meas_pars[0],
        mean_meas=meas_pars[1],
        var_meas=meas_pars[2],
    )
    return state_filt, logdens


def _process_times(obs_times, t_min, t_max, n_steps):
    """
    Process observation times.
    """
    sim_times = jnp.linspace(t_min, t_max, n_steps + 1)
    obs_ind = jnp.searchsorted(sim_times, obs_times)
    has_obs = jnp.full_like(sim_times, fill_value=False, dtype=bool)
    has_obs = has_obs.at[obs_ind].set(True)
    return sim_times, obs_ind, has_obs
