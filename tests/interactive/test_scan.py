# --- tests for new rodeo functionality ----------------------------------------

# from `rodeo` root folder:
# uv run python -m tests.interactive.test_scan


# --- Import libraries and modules ---------------------------------------------

# import inspect
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import rodeo  # ODE solvers
import rodeo.interrogate  # interrogation methods
import rodeo.kalmantv.standard
import rodeo.prior  # IBM prior
import tests.interactive.dalton
import tests.interactive.fenrir
import tests.interactive.solve
import tests.interactive.utils
import tests.utils
# helper function to help with the padding in rodeo format
from rodeo.utils import first_order_pad

jax.config.update("jax_enable_x64", True)


# --- helpers ------------------------------------------------------------------


def get_sim_grid(dt_max, t_range):
    """
    Compute the grid used for simulating the ODE.

    Args:
        dt_max (float) : Maximum grid size,
        t_range (float) : Time range of simulation `t_max - t_min`.

    Returns:
        dt_sim (float) : Simulation interobservation time.
        n_steps (int) : Number of time steps, such that `t_range = dt_sim * n_steps`.
    """
    n_steps = int(jnp.ceil(t_range / dt_max))
    dt_sim = t_range / n_steps
    return dt_sim, n_steps


def get_obs_ind(sim_times, obs_times):
    """
    Obtain indices of `sim_times` corresponding to `obs_times`.
    """
    return jnp.searchsorted(a=sim_times, v=obs_times)


# --- Kalman stuff -------------------------------------------------------------


class KalmanAlgorithms(NamedTuple):
    """
    A container for Kalman algorithms.

    Warning: `rodeo.kalmantv.standard` has a `_smooth` function which is not
        declared here...
    """

    predict: Callable | None = None
    update: Callable | None = None
    filter: Callable | None = None
    smooth_mv: Callable | None = None
    smooth_sim: Callable | None = None
    smooth_cond: Callable | None = None
    smooth: Callable | None = None
    forecast: Callable | None = None
    multivariate_normal_logpdf: Callable | None = None
    multivariate_normal_sim: Callable | None = None


def make_kalman(kalman, multivariate_normal_logpdf=None):
    """
    Convert an arbitrary Kalman container to an instance of `KalmanAlgorithms`.
    """
    kalman_dict = {}
    kalman_keys = [
        "predict",
        "update",
        "filter",
        "smooth_mv",
        "smooth_sim",
        "smooth_cond",
        "smooth",
        "forecast",
    ]
    for k in kalman_keys:
        kalman_dict[k] = getattr(kalman, k, None)

    kalman_dict["multivariate_normal_logpdf"] = multivariate_normal_logpdf

    return KalmanAlgorithms(**kalman_dict)


def make_kalman_diag(kalman, multivariate_normal_sim=None):
    """
    Diagonalize a `KalmanAlgorithms` object.

    Apply `jax.vmap()` along the first dimension of each argument of each method.

    Warning: Not sure this is what we want for `*args` and `**kwargs`...
    """
    kalman_dict = {}
    kalman_keys = [
        "predict",
        "update",
        "filter",
        "smooth_mv",
        "smooth_sim",
        "smooth_cond",
        "smooth",
        "forecast",
    ]
    for k in kalman_keys:
        f = getattr(kalman, k, None)
        if f is not None:
            kalman_dict[k] = jax.vmap(f)
            # seems this is not needed
            # kalman_dict[k].__signature__ = inspect.signature(f)

    if getattr(kalman, "multivariate_normal_logpdf", None) is not None:

        f = getattr(kalman, "multivariate_normal_logpdf")

        def multivariate_normal_logpdf(x, mean, cov):
            lp = jax.vmap(f)(x=x, mean=mean, cov=cov)
            return jnp.sum(lp)

        kalman_dict["multivariate_normal_logpdf"] = multivariate_normal_logpdf

    if multivariate_normal_sim is not None:
        kalman_dict["multivariate_normal_sim"] = multivariate_normal_sim
    else:
        if getattr(kalman, "multivariate_normal_sim", None) is not None:

            f = getattr(kalman, "multivariate_normal_sim", None)

            def multivariate_normal_sim(key, mean, cov):
                n_dim = jax.tree.leaves(mean)[0].shape[0]
                keys = jax.random.split(key, num=n_dim)
                return jax.vmap(f)(key=keys, mean=mean, cov=cov)

        kalman_dict["multivariate_normal_sim"] = multivariate_normal_sim

    return KalmanAlgorithms(**kalman_dict)


kalman_standard = make_kalman(
    kalman=rodeo.kalmantv.standard,
    multivariate_normal_logpdf=tests.interactive.utils.multivariate_normal_logpdf,
)
kalman_standard_diag = make_kalman_diag(kalman=kalman_standard)

# --- test multivariate_normal_logpdf ------------------------------------------

n_var = 6
n_rm = 3
n_eq = 1

key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, num=3)
a = jax.random.normal(subkeys[0], shape=(n_var - n_rm, n_var - n_rm))
a = a @ a.T
if n_rm > 0:
    key, subkey = jax.random.split(key)
    keep_ind = jax.random.choice(
        subkey,
        a=jnp.arange(n_var),
        shape=(n_var - n_rm,),
        replace=False,
    )
    keep_ind = jnp.sort(keep_ind)
    rm_bool = jnp.full(shape=(n_var,), fill_value=True).at[keep_ind].set(False)
    A = jnp.zeros(shape=(n_var, n_var))
    A = A.at[jnp.ix_(keep_ind, keep_ind)].set(a)
    A = A.at[rm_bool, rm_bool].set(0.0)  # no longer using jnp.inf
else:
    A = a
b = jax.random.normal(subkeys[1], shape=(n_var, n_eq))

logpdf = jax.scipy.stats.multivariate_normal.logpdf(
    x=jnp.squeeze(b)[keep_ind],
    mean=jnp.zeros((n_var - n_rm,)),
    cov=a,
)
logpdf2 = tests.interactive.utils.multivariate_normal_logpdf(
    x=jnp.squeeze(b)[keep_ind],
    mean=jnp.zeros((n_var - n_rm,)),
    cov=a,
)
logpdf3 = tests.interactive.utils.multivariate_normal_logpdf(
    x=jnp.squeeze(b),
    mean=jnp.zeros((n_var,)),
    cov=A,
)

ans = {
    "scipy_logpdf_a": logpdf,
    "mvn_logpdf_a": logpdf2,
    "mvn_logpdf_A": logpdf3,
}
print(ans)

# --- Define the ODE-IVP -------------------------------------------------------


def fitz_fun(X, t, theta):
    r"""
    FitzHugh-Nagumo ODE in rodeo format.

    Note: Made the model time-dependent for testing.

    Args:
        X (jax.ArrayLike): Array of `(n_vars, n_deriv)`, where in this case `n_var = 2`.
        t (float): Time variable.
        theta (jax.Array): Model parameters.

    Returns:
        (jax.Array): Array of shape `(n_var, 1)`.
    """
    a, b, c = theta
    V, R = X[:, 0]
    return jnp.array(
        [
            [c * (V - V * V * V / 3 + R)],
            [-1 / c * (V - a + b * R)],
        ]
    ) / (t + 1.0)


# time interval on which a solution is sought
t_min = 0.0
t_max = 40.0

x0 = jnp.array([-1.0, 1.0])  # initial value
theta = jnp.array([0.2, 0.2, 3])  # ODE model parameters

# --- test basic_solve ---------------------------------------------------------

n_vars = x0.shape[0]  # number of ODE variables
n_deriv = 3  # number of derivatives to use

# simulation interobservation time and number of simulation steps
dt_sim, n_steps = get_sim_grid(dt_max=5.0, t_range=t_max - t_min)

# IBM process scale factors
sigma = jnp.array([0.1] * n_vars)

# zero-padding
# FIXME: first_order_pad could pad ODE function as well...
W, fitz_init = first_order_pad(ode_fun=fitz_fun, n_vars=n_vars, n_deriv=n_deriv)
W  # zero-padded weight matrix
X0 = fitz_init(x0, t=t_min, theta=theta)  # initial value in rodeo format

# GP prior parameters
prior_pars = rodeo.prior.ibm_init(dt=dt_sim, n_deriv=n_deriv, sigma=sigma)


key = jax.random.PRNGKey(0)

# basic: filter
key, subkey = jax.random.split(key)
filter_out = rodeo.solve._solve_filter(
    key=subkey,
    # define ode
    ode_fun=fitz_fun,
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    theta=theta,  # ODE parameters added here
    # solver parameters
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_weight=prior_pars[0],
    prior_var=prior_pars[1],
    kalman_funs=rodeo.kalmantv.standard,  # default choice of Kalman filter
)

filter_out2 = tests.interactive.solve.solve_filter(
    key=subkey,
    # define ode
    ode_fun=lambda X, t: fitz_fun(X, t, theta),
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    # theta=theta,  # ODE parameters added here
    # solver parameters
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    kalman_funs=kalman_standard_diag,  # default choice of Kalman filter
)

ans = jax.tree.map(lambda x, y: tests.utils.rel_err(x, y), filter_out, filter_out2)
print(f"basic_filter: {ans}")

# basic: solve_mv
key, subkey = jax.random.split(key)
mv_out = tests.interactive.solve.solve_mv(
    key=subkey,
    # define ode
    ode_fun=lambda X, t: fitz_fun(X, t, theta),
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    # theta=theta,  # ODE parameters added here
    # solver parameters
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    kalman_funs=kalman_standard_diag,  # default choice of Kalman filter
)


mv_out2 = rodeo.solve_mv(
    key=subkey,
    # define ode
    ode_fun=fitz_fun,
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    theta=theta,  # ODE parameters added here
    # solver parameters
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    kalman_type="standard",  # default choice of Kalman filter
)

ans = jax.tree.map(lambda x, y: tests.utils.rel_err(x, y), mv_out, mv_out2)
print(f"basic_solve_mv: {ans}")

# basic: solve_sim
key, subkey = jax.random.split(key)
sim_out = tests.interactive.solve.solve_sim(
    key=subkey,
    # define ode
    ode_fun=lambda X, t: fitz_fun(X, t, theta),
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    # theta=theta,  # ODE parameters added here
    # solver parameters
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    kalman_funs=kalman_standard_diag,  # default choice of Kalman filter
)


sim_out2 = rodeo.solve_sim(
    key=subkey,
    # define ode
    ode_fun=fitz_fun,
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    theta=theta,  # ODE parameters added here
    # solver parameters
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    kalman_type="standard",  # default choice of Kalman filter
)

ans = jax.tree.map(lambda x, y: tests.utils.rel_err(x, y), sim_out, sim_out2)
print(f"basic_solve_sim: {ans}")

# --- test: fenrir -------------------------------------------------------------

n_vars = x0.shape[0]  # number of ODE variables
n_deriv = 3  # number of derivatives to use

# simulation interobservation time and number of simulation steps
dt_sim, n_steps = get_sim_grid(dt_max=5.0, t_range=t_max - t_min)
sim_times = jnp.linspace(t_min, t_max, n_steps + 1)

# IBM process scale factors
sigma = jnp.array([0.1] * n_vars)

# zero-padding
# FIXME: first_order_pad could pad ODE function as well...
W, fitz_init = first_order_pad(ode_fun=fitz_fun, n_vars=n_vars, n_deriv=n_deriv)
W  # zero-padded weight matrix
X0 = fitz_init(x0, t=t_min, theta=theta)  # initial value in rodeo format

# GP prior parameters
prior_pars = rodeo.prior.ibm_init(dt=dt_sim, n_deriv=n_deriv, sigma=sigma)


# simulate observations
dt_obs = 10.0  # interobservation time
n_obs = int((t_max - t_min) / dt_obs) + 1
# observation times
obs_times = jnp.linspace(t_min, t_max, num=n_obs)
obs_sd = 0.2  # standard deviation in noise model

# generate ODE solution
Xt, _ = rodeo.solve_mv(
    key=None,
    # define ode
    ode_fun=fitz_fun,
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    theta=theta,  # ODE parameters added here
    # solver parameters
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    kalman_type="standard",  # default choice of Kalman filter
)

# generate observations
obs_ind = get_obs_ind(sim_times=sim_times, obs_times=obs_times)
eps = jax.random.normal(key=subkey, shape=(n_obs, 2))
obs_data = Xt[obs_ind, :, 0] + obs_sd * eps

# convert to rodeo format
n_meas_per_var = 1  # number of measurements per variable in obs_data_i
obs_data = jnp.expand_dims(obs_data, axis=-1)  # shape = (n_obs, n_vars, n_meas_per_var)
# first-order form, one observation: shape = (n_vars, n_meas_per_var)
obs_weight = jnp.array([[1.0], [1.0]])
# pad for full form, one observation: shape = (n_vars, n_meas_per_var, n_deriv)
obs_weight = jnp.concatenate(
    [
        jnp.expand_dims(obs_weight, -1),
        jnp.zeros((n_vars, n_meas_per_var, n_deriv - 1)),
    ],
    axis=-1,
)
# full form, all observations: shape = (n_obs, n_vars, n_meas_per_var, n_deriv)
obs_weight = jnp.array(n_obs * [obs_weight])
# full form, one observation: shape = (n_vars, n_meas_per_var, n_meas_per_var)
obs_var = obs_sd**2 * jnp.ones((n_vars, n_meas_per_var, n_meas_per_var))
# full form, all observations: shape = (n_obs, n_vars, n_meas_per_var, n_meas_pre_var)
obs_var = jnp.array(n_obs * [obs_var])

key, subkey = jax.random.split(key)
fenrir_out = rodeo.inference.fenrir(
    key=subkey,  # immaterial, since not used
    # ode specification
    ode_fun=fitz_fun,
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    theta=theta,
    # solver
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    # gaussian measurement model
    obs_data=obs_data,
    obs_times=obs_times,
    obs_weight=obs_weight,
    obs_var=obs_var,
    kalman_type="standard",
)

fenrir_out2 = tests.interactive.fenrir.fenrir(
    key=subkey,  # immaterial, since not used
    # ode specification
    ode_fun=lambda X, t: fitz_fun(X, t, theta),
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    # theta=theta,
    # solver
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    # gaussian measurement model
    obs_data=obs_data,
    obs_times=obs_times,
    obs_weight=obs_weight,
    obs_var=obs_var,
    kalman_funs=kalman_standard_diag,  # default choice of Kalman filter
)

ans = jax.tree.map(lambda x, y: tests.utils.rel_err(x, y), fenrir_out, fenrir_out2[0])
print(f"fenrir_logpdf: {ans}")


# --- test: dalton -------------------------------------------------------------

key, subkey = jax.random.split(key)
dalton_out = rodeo.inference.dalton(
    key=subkey,  # immaterial, since not used
    # ode specification
    ode_fun=fitz_fun,
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    theta=theta,
    # solver
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    # gaussian measurement model
    obs_data=obs_data,
    obs_times=obs_times,
    obs_weight=obs_weight,
    obs_var=obs_var,
    kalman_type="standard",
)


dalton_out2 = tests.interactive.dalton.dalton(
    key=subkey,  # immaterial, since not used
    # ode specification
    ode_fun=lambda X, t: fitz_fun(X, t, theta),
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    # solver
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    # gaussian measurement model
    obs_data=obs_data,
    obs_times=obs_times,
    obs_weight=obs_weight,
    obs_var=obs_var,
    kalman_funs=kalman_standard_diag,  # default choice of Kalman filter
)

ans = jax.tree.map(lambda x, y: tests.utils.rel_err(x, y), dalton_out, dalton_out2)
print(f"dalton_logpdf: {ans}")


# --- test: daltonng -----------------------------------------------------------


def fitz_loglik_i(obs_data, ode_data, i_obs, theta=None):
    obs_weight = jnp.array([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    obs_var = obs_sd**2 * jnp.ones((2, 1, 1))
    lp = jax.vmap(
        lambda y, x, w, v: jax.scipy.stats.multivariate_normal.logpdf(
            x=y, mean=jnp.dot(w, x), cov=v
        )
    )(obs_data, ode_data, obs_weight, obs_var)
    return jnp.sum(lp)


i_obs = 1
daltonng_loglik = fitz_loglik_i(obs_data[i_obs], Xt[i_obs], i_obs)
daltonng_loglik2 = jnp.sum(
    jax.scipy.stats.norm.logpdf(
        x=obs_data[i_obs, :, 0], loc=Xt[i_obs, :, 0], scale=obs_sd
    )
)

ans = jax.tree.map(
    lambda x, y: tests.utils.rel_err(x, y), daltonng_loglik, daltonng_loglik2
)
print(f"daltonng_obs_loglik_i: {ans}")


key, subkey = jax.random.split(key)
daltonng_out = rodeo.inference.daltonng(
    key=subkey,  # immaterial, since not used
    # ode specification
    ode_fun=fitz_fun,
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    theta=theta,
    # solver
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    # non-gaussian measurement model
    obs_data=obs_data,
    obs_times=obs_times,
    obs_loglik_i=fitz_loglik_i,
    kalman_type="standard",
)


daltonng_out2 = tests.interactive.dalton.daltonng(
    key=subkey,  # immaterial, since not used
    # ode specification
    ode_fun=lambda X, t: fitz_fun(X, t, theta),
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    # solver
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    # non-gaussian measurement model
    obs_data=obs_data,
    obs_times=obs_times,
    obs_loglik_i=fitz_loglik_i,
    kalman_funs=kalman_standard_diag,  # default choice of Kalman filter
)


breakpoint()


ans = jax.tree.map(
    lambda x, y: tests.utils.rel_err(x[0], y[0]), daltonng_out, daltonng_out2
)
print(f"daltonng_out: {ans}")


def seirah_fun(X, t, theta):
    """
    SEIRAH ODE for Covid-19.

    Args:
        X (ArrayLike): Array of shape `(n_vars, n_deriv)` where `n_vars = 6`.
        t (float): Time variable
        theta (ArrayLike): Model parameters.

    Returns:
        (Array): dX/dt, an array of shape `(n_vars, 1)`.
    """
    S, E, I, R, A, H = X[:, 0]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b * S * (I + alpha * A) / N
    dE = b * S * (I + alpha * A) / N - E / D_e
    dI = r * E / D_e - I / D_q - I / D_I
    dR = (I + A) / D_I + H / D_h
    dA = (1 - r) * E / D_e - A / D_I
    dH = I / D_q - H / D_h
    return jnp.array([[dS], [dE], [dI], [dR], [dA], [dH]])


# time interval on which a solution is sought
t_min = 0.0
t_max = 60.0

x0 = jax.random.exponential(key, shape=(6,)) * 500000.0  # initial value
theta = jnp.array([0.15, 0.1, 0.25, 14.0, 27.0, 150.0])  # ODE model parameters

# prior
n_vars = 6  # number of ODE variables
n_deriv = 3  # number of derivatives to use

# simulation interobservation time and number of simulation steps
dt_max = 0.5
dt_sim, n_steps = get_sim_grid(dt_max=dt_max, t_range=t_max - t_min)

# IBM process scale factors
sigma = jnp.array([0.1] * n_vars)

# zero-padding
W, seirah_init = first_order_pad(ode_fun=seirah_fun, n_vars=n_vars, n_deriv=n_deriv)
X0 = seirah_init(x0, t=t_min, theta=theta)  # initial value in rodeo format

# GP prior parameters
prior_pars = rodeo.prior.ibm_init(dt=dt_sim, n_deriv=n_deriv, sigma=sigma)

# ODE solution
Xt, _ = rodeo.solve.solve_mv(
    key=None,
    # define ODE
    ode_fun=seirah_fun,
    ode_weight=W,
    ode_init=X0,
    t_min=t_min,
    t_max=t_max,
    theta=theta,  # ODE parameters
    # define solver
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_pars=prior_pars,
    kalman_type="standard",  # default choice of Kalman filter
)


# generate observations
dt_obs = 1.0  # interobservation time
n_obs = int((t_max - t_min) / dt_obs) + 1
# observation times
obs_times = jnp.linspace(t_min, t_max, num=n_obs)
_, obs_ind, _ = tests.interactive.utils._process_times(obs_times, t_min, t_max, n_steps)


def seirah_obs_pars(X, theta):
    """
    Parameters of the SEIRAH model likelihood.

    Returns:
        (Array): New incident cases of latent and hospitalized compartments,
            which are the means of the Poisson variables observed at that time.
    """
    b, r, alpha, D_e, D_I, D_q = theta
    I_in = r * X[1, 0] / D_e
    H_in = X[2, 0] / D_q
    return jnp.array([I_in, H_in])


key, *subkeys = jax.random.split(key, num=n_obs + 1)
subkeys = jnp.array(subkeys)
obs_data = jax.vmap(
    lambda x, k: jax.random.poisson(key=k, lam=seirah_obs_pars(x, theta))
)(Xt[obs_ind], subkeys)
obs_data = jnp.expand_dims(obs_data, axis=-1)


def seirah_obs_loglik_i(obs_data_i, ode_data_i, ind, theta):
    """
    Likelihood function for the SEIRAH model in non-Gaussian DALTON format.
    """
    Xin = seirah_obs_pars(ode_data_i, theta)
    return jnp.sum(jax.scipy.stats.poisson.logpmf(obs_data_i.flatten(), Xin))


# --- kalman test --------------------------------------------------------------


def full_inf(shape):
    return jnp.full(shape=shape, fill_value=jnp.inf)


gain_factor = jnp.array(
    [[0.11727148, 0.10642587, 0.04782057], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
)

var_innov = jnp.array([[0.15727148, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

var_innov = jax.random.normal(key, shape=(5, 5))
var_innov = jax.scipy.linalg.block_diag(
    jnp.dot(var_innov, var_innov.T), jnp.diag(full_inf((3,)))
)

vals, vecs = jnp.linalg.eigh(var_innov)
z = jnp.dot(vecs.T, gain_factor)
not_zero = ~jnp.isclose(vals, 0.0, rtol=1e-300)
new_vals = jnp.where(not_zero, vals, 1.0)
inv_vals = jnp.where(not_zero, 1.0 / new_vals, 0.0)
sol = jnp.linalg.multi_dot([vecs, jnp.diag(inv_vals), z])

state_pred = (
    jnp.array(
        [[-12.90225185, -6.06101044, -0.90991025], [0.33384588, 0.59660063, 0.20790671]]
    ),
    jnp.array(
        [
            [
                [0.11727148, 0.10642587, 0.04782057],
                [0.10642587, 0.10070437, 0.04973755],
                [0.04782057, 0.04973755, 0.03207709],
            ],
            [
                [0.03380566, 0.03941548, 0.02303507],
                [0.03941548, 0.04992398, 0.03161432],
                [0.02303507, 0.03161432, 0.02580003],
            ],
        ]
    ),
)

x_pseudo = jnp.array(
    [
        [-8.75063166, 0.0, 0.0],
        [0.12404529, 0.0, 0.0],
    ]
)

pseudo_pars = (
    jnp.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    ),
    jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    jnp.array(
        [
            [[0.04, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.04, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    ),
)

# x_meas = jnp.array([[0.0], [0.0]])

# meas_pars = (
#     jnp.array(
#         [
#             [[4.51276644e01, 1.00000000e00, 0.00000000e00]],
#             [[6.06060606e-03, 1.00000000e00, 0.00000000e00]],
#         ]
#     ),
#     jnp.array([[390.42047572], [-0.39703793]]),
#     jnp.array([[[0.0]], [[0.0]]]),
# )

breakpoint()


kalman_standard_diag.update(
    mean_state_pred=state_pred[0],
    var_state_pred=state_pred[1],
    x_meas=x_pseudo,
    wgt_meas=pseudo_pars[0],
    mean_meas=pseudo_pars[1],
    var_meas=pseudo_pars[2],
)


# --- Import libraries and modules ---------------------------------------

import jax
import jax.numpy as jnp
import jaxopt  # optimization library for finding mode in Laplace approximation
import rodeo.inference  # parameter inference methods
import rodeo.interrogate  # interrogation methods
import rodeo.prior  # IBM prior

# --- parameter inference: basic + laplace -------------------------------


def fitz_logprior(upars):
    "Logprior on unconstrained model parameters."
    n_theta = 5  # number of ODE + IV parameters
    lpi = jax.scipy.stats.norm.logpdf(x=upars[:n_theta], loc=0.0, scale=10.0)
    return jnp.sum(lpi)


def fitz_loglik(obs_data, ode_data, theta):
    """
    Loglikelihood for measurement model.

    Args:
        obs_data (ndarray(n_obs, n_vars)): Observations data.
        ode_data (ndarray(n_obs, n_vars, n_deriv)): ODE solution.
    """
    ll = jax.scipy.stats.norm.logpdf(x=obs_data, loc=ode_data[:, :, 0], scale=noise_sd)
    return jnp.sum(ll)


def fitz_constrain_pars(upars, dt):
    """
    Convert unconstrained optimization parameters into rodeo inputs.

    Args:
        upars : Parameters vector on unconstrainted scale.
        dt : Discretization grid size.

    Returns:
        tuple with elements:
        - theta : ODE parameters.
        - X0 : Initial values in rodeo format.
        - Q, R : Prior matrices.
    """
    theta = jnp.exp(upars[:3])
    x0 = upars[3:5]
    X0 = fitz_init_pad(x0, 0, theta=theta)
    sigma = jnp.exp(upars[5:])
    Q, R = rodeo.prior.ibm_init(dt=dt, n_deriv=n_deriv, sigma=sigma)
    return theta, X0, Q, R


def fitz_laplace(key, neglogpost, n_samples, upars_init):
    """
    Sample from the Laplace approximation to the parameter posterior for the FN model.

    Args:
        key : PRNG key.
        neglogpost: Function specifying the negative log-posterior distribution
                    in terms of the unconstrained parameter vector ``upars``.
        upars_init: Initial value to the optimization algorithm over ``neglogpost()``.
        n_samples : Number of posterior samples to draw.

    Returns:
        JAX array of shape ``(n_samples, 5)`` of posterior samples from ``(theta, x0)``.
    """
    n_theta = 5  # number of ODE + IV parameters
    # find mode of neglogpost()
    solver = jaxopt.ScipyMinimize(
        fun=neglogpost,
        method="Newton-CG",
        jit=True,  # jaxopt jits the neglogpost function for us
    )
    opt_res = solver.run(upars_init)
    upars_mean = opt_res.params
    # variance estimate
    upars_fisher = jax.jacfwd(jax.jacrev(neglogpost))(upars_mean)
    # unconstrained ode+iv parameter variance estimate
    uode_var = jax.scipy.linalg.inv(upars_fisher[:n_theta, :n_theta])
    uode_mean = upars_mean[:n_theta]
    # sample from Laplace approximation
    uode_sample = jax.random.multivariate_normal(
        key=key, mean=uode_mean, cov=uode_var, shape=(n_samples,)
    )
    # convert back to original scale
    ode_sample = uode_sample.at[:, :3].set(jnp.exp(uode_sample[:, :3]))
    return ode_sample


def fitz_neglogpost_basic(upars):
    "Negative logposterior for basic approximation."
    # solve ODE
    theta, X0, prior_Q, prior_R = fitz_constrain_pars(upars, dt_sim)
    # basic loglikelihood
    ll, Xt = rodeo.inference.basic(
        key=None,  # immaterial, since not used
        # ode specification
        ode_fun=fitz_fun,
        ode_weight=W,
        ode_init=X0,
        t_min=t_min,
        t_max=t_max,
        theta=theta,
        # solver parameters
        n_steps=n_steps,
        interrogate=rodeo.interrogate.interrogate_kramer,
        prior_weight=prior_Q,
        prior_var=prior_R,
        # observations
        obs_data=Y,
        obs_times=obs_times,
        obs_loglik=fitz_loglik,
        kalman_type="standard",
    )
    return -(ll + fitz_logprior(upars))


# keys
basic_key, fenrir_key, hmc_key, mcmc_key, magi_key = jax.random.split(key, num=5)

# starting values for optimization
upars_init = jnp.append(jnp.ones(len(theta)), X0[:, 0])
upars_init = jnp.append(upars_init, jnp.log(0.01 * jnp.ones(n_vars)))
# samples for posterior
n_samples = 100_000

# Laplace approximation via basic likelihood
Theta_basic = fitz_laplace(basic_key, fitz_neglogpost_basic, n_samples, upars_init)

# --- Import libraries and modules ---------------------------------------

import blackjax  # mcmc algorithms written in jax
import jax
import jax.numpy as jnp

# --- parameter inference: basic + hmc -----------------------------------


# blackjax uses the loglikelihood function instead of the negative
def fitz_logpost_basic(upars):
    return -fitz_neglogpost_basic(upars)


def fitz_hmc(key, logpost, n_samples, upars_init, num_steps):
    """
    Sample from the parameter posterior via the Hamiltonian Monte Carlo
    method for the FN model.

    Args:
        key : PRNG key.
        logpost : Function specifying the log-posterior distribution
                  n terms of the unconstrained parameter vector ``upars``.
        upars_init : Initial value to the optimization algorithm over ``logpost()``.
        n_samples : Number of posterior samples to draw.
        num_steps : Number of HMC leapfrog steps.

    Returns:
        JAX array of shape ``(n_samples, 5)`` of posterior samples from ``(theta, x0)``.
    """
    key, *subkeys = jax.random.split(key, num=3)
    # Stan window adaptation algorithm to start with reasonable parameters
    warmup = blackjax.window_adaptation(
        blackjax.hmc, logpost, num_integration_steps=num_steps
    )
    (initial_state, parameters), _ = warmup.run(subkeys[0], upars_init)
    kernel = blackjax.hmc(logpost, **parameters).step

    # standard in blackjax to write an inference loop for sampling
    def inference_loop(key, kernel, initial_state, n_samples):

        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(key, n_samples)
        # scan will automatically jit compile for you so no need to do so explicitly
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states

    uode_sample = inference_loop(subkeys[1], kernel, initial_state, n_samples).position
    if isinstance(uode_sample, dict):
        uode_sample = uode_sample["upars"]
    else:
        uode_sample = uode_sample[:, :5]
    # convert back to original scale
    ode_sample = uode_sample.at[:, :3].set(jnp.exp(uode_sample[:, :3]))
    return ode_sample


# HMC via basic likelihood
Theta_hmc = fitz_hmc(hmc_key, fitz_logpost_basic, n_samples, upars_init, num_steps=5)

# --- Import libraries and modules -------------------------------------------

import jax
import jax.numpy as jnp
import jaxopt  # optimization library for finding mode in Laplace approximation
import rodeo.inference  # parameter inference methods
import rodeo.interrogate  # interrogation methods

# --- parameter inference: fenrir + laplace ----------------------------------

# gaussian measurement model specification in blocked form
n_meas = 1  # number of measurements per variable in obs_data_i
obs_data = jnp.expand_dims(Y, axis=-1)
obs_weight = jnp.zeros((len(obs_data), n_vars, n_meas, n_deriv))
obs_weight = obs_weight.at[:].set(jnp.array([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]))
obs_var = jnp.zeros((len(obs_data), n_vars, n_meas, n_meas))
obs_var = obs_var.at[:].set(noise_sd**2 * jnp.array([[[1.0]], [[1.0]]]))


def fitz_neglogpost_fenrir(upars):
    "Negative logposterior for basic approximation."
    theta, X0, prior_Q, prior_R = fitz_constrain_pars(upars, dt_sim)
    # fenrir loglikelihood
    ll = rodeo.inference.fenrir(
        key=None,  # immaterial, since not used
        # ode specification
        ode_fun=fitz_fun,
        ode_weight=W,
        ode_init=X0,
        t_min=t_min,
        t_max=t_max,
        theta=theta,
        # solver
        n_steps=n_steps,
        interrogate=rodeo.interrogate.interrogate_kramer,
        prior_weight=prior_Q,
        prior_var=prior_R,
        # gaussian measurement model
        obs_data=obs_data,
        obs_times=obs_times,
        obs_weight=obs_weight,
        obs_var=obs_var,
        kalman_type="standard",
    )
    return -(ll + fitz_logprior(upars))


# Laplace approximation via fenrir likelihood
Theta_fenrir = fitz_laplace(fenrir_key, fitz_neglogpost_fenrir, n_samples, upars_init)

# --- Import libraries and modules ---------------------------------------

import functools  # partial to fix kalman_type

import jax
import jax.numpy as jnp
import rodeo  # ODE solvers
import rodeo.inference  # parameter inference methods
import rodeo.interrogate  # interrogation methods
from rodeo.inference import \
    random_walk_aux  # random walk kernel for marginal mcmc

# --- parameter inference: chkrebtii -------------------------------------

# fitx the value for kalman_type
interrogate_chkbretii_partial = functools.partial(
    rodeo.interrogate.interrogate_chkrebtii, kalman_type="standard"
)


def fitz_logpost_mcmc(key, upars):
    """
    Computes marginal MCMC posterior.

    Also returns solution path Xt that was generated.
    """
    theta, X0, prior_Q, prior_R = fitz_constrain_pars(upars, dt_sim)
    Xt = rodeo.solve_sim(
        key=key,
        # define ode
        ode_fun=fitz_fun,
        ode_weight=W,
        ode_init=X0,
        t_min=t_min,
        t_max=t_max,
        theta=theta,
        # solver parameters
        n_steps=n_steps,
        interrogate=interrogate_chkbretii_partial,
        prior_weight=prior_Q,
        prior_var=prior_R,
        kalman_type="standard",
    )
    ode_data = Xt[obs_ind]
    lp = fitz_logprior(upars) + fitz_loglik(Y, ode_data)
    return lp, Xt


def fitz_mcmc(key, n_samples, upars):
    r"""
    Marginal MCMC using blackjax.
    """
    key, subkey = jax.random.split(key)
    # initialize mcmc
    state = random_walk_aux.init(
        position=upars, logdensity_fn=lambda upars: fitz_logpost_mcmc(subkey, upars)
    )
    keys = jax.random.split(key, num=n_samples)
    # blackjax
    # RW with auxiliary outputs
    rwa_kernel = random_walk_aux.build_additive_step()
    # standard deviation of the random walk
    scale = jnp.array([0.01, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01])

    # inference loop
    def marginal_rw_step(key, state, sigma):
        "One step of Marginal Random Walk."
        keys = jax.random.split(key, num=2)
        return rwa_kernel(
            rng_key=keys[0],  # for RW proposal
            state=state,
            # logpost for each step has its own key for generating Xt
            logdensity_fn=lambda position: fitz_logpost_mcmc(keys[1], position),
            random_step=blackjax.mcmc.random_walk.normal(sigma=sigma),
        )

    def step(state, rng_key):
        state, info = marginal_rw_step(key=rng_key, state=state, sigma=scale)
        return state, (state, info)

    # scan will automatically jit compile for you so no need to do so explicitly
    _, out = jax.lax.scan(f=step, init=state, xs=keys)
    uode_sample = out[0].position[:, :5]
    # convert back to original scale
    ode_sample = uode_sample.at[:, :3].set(jnp.exp(uode_sample[:, :3]))
    return ode_sample


# Chkrebtii marginal MCMC
Theta_mcmc = fitz_mcmc(mcmc_key, n_samples, upars_init)

# --- Import libraries and modules ---------------------------------------

import blackjax  # mcmc algorithms written in jax
import jax
import jax.numpy as jnp
import rodeo.inference  # parameter inference methods
import rodeo.prior  # IBM prior

# --- parameter inference: magi ------------------------------------------


def fitz_expand(X, theta):
    """
    Expand FN data.
    """
    return jax.vmap(
        fun=lambda x, t: fitz_init_pad(x, t, theta=theta),
    )(X, sim_times)


def fitz_magi_logpost_fn(upars, ode_data_subset):
    """
    Joint posterior function for the MAGI approximation.
    """
    theta = jnp.exp(upars[:3])
    x0 = upars[3:5]
    ode_data_subset = ode_data_subset.at[0].set(x0)
    beta = (
        dt_obs * dt_sim ** (2 - 2 * n_deriv) * sigma[0] ** (-2)
    )  # magi prior temperature
    lp = (
        rodeo.inference.magi_logdens(
            ode_data_subset=ode_data_subset,
            ode_expand=fitz_expand,
            n_active=2,  # 0th and 1st order derivatives
            prior_weight=prior_Q,
            prior_var=prior_Rhalf,
            theta=theta,
            # use square-root version for numerical stability
            kalman_type="square-root",
        )
        / beta
    )
    sim_times = jnp.linspace(t_min, t_max, n_steps + 1)
    ode_data = ode_data_subset[jnp.searchsorted(sim_times, obs_times)][:, :, None]
    lp += fitz_loglik(Y, ode_data) + fitz_logprior(upars)
    return lp


fitz_magi_logpost = lambda x: fitz_magi_logpost_fn(**x)
sigma = jnp.array([0.1] * n_vars)  # fixed at arbitrary value
prior_Q, prior_R = rodeo.prior.ibm_init(dt=dt_sim, n_deriv=n_deriv, sigma=sigma)
prior_Rhalf = jax.vmap(jnp.linalg.cholesky)(prior_R)
upars_init = jnp.append(jnp.ones(len(theta)), X0[:, 0])
# linear interpolation as an initial guess for MCMC
ode_subset_init = jax.vmap(jnp.interp, in_axes=[None, None, 1])(
    sim_times, obs_times, Y
).T
magi_upars_init = {"upars": upars_init, "ode_data_subset": ode_subset_init}
Theta_magi = fitz_hmc(
    magi_key, fitz_magi_logpost, n_samples, magi_upars_init, num_steps=200
)


# --- scratch ------------------------------------------------------------------


def foo(x, flag):
    if flag:
        y = x + 2
    else:
        y = x**2
    return y, y


def bar(x, flag):
    return jax.lax.scan(f=foo, init=x, xs=flag)


x = jnp.array(1.0)
flag = (True, True, False, False)

jax.jit(bar, static_argnums=1)(x, flag)


def make_fitz_setup(x0, t_min, t_max, dt_sim):
    """
    Args:
    """
    n_vars = 2
    W, fitz_init_pad = first_order_pad(fitz_fun, n_vars=n_vars, n_deriv=n_deriv)
    # FIXME: make dt_sim a multiple of t_range

    def setup(theta, sigma):
        # initial value in rodeo format
        # prior parameters
        X0 = fitz_init_pad(x0, t=t_min, theta=theta)
        prior_pars = rodeo.prior.ibm_init(dt=dt_sim, n_deriv=n_deriv, sigma=sigma)

        def fun(X, t):
            return fitz_fun(X=X, t=t, theta=theta)


def _solve_filter(
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

    # dimensions of block, meas, and state variables
    n_block, n_bmeas, n_bstate = ode_weight.shape

    # initial values
    mean_state_init = ode_init
    var_state_init = jnp.zeros((n_block, n_bstate, n_bstate))
    state_init = (mean_state_init, var_state_init)

    # observed data
    obs = {"t": jnp.linspace(t_min, t_max, num=n_steps + 1)[:-1]}
    if key is not None:
        obs["key"] = jax.random.split(key, num=n_steps)
    else:
        obs["key"] = jnp.zeros(n_steps)

    # scan functions
    x_meas = jnp.zeros((n_block, n_bmeas))
    mean_state = jnp.zeros((n_block, n_bstate))
    prior_weight, prior_var = prior_pars

    def predict_fn(state_past, x, kalman_funs):
        return {
            "mean_state": mean_state,
            "wgt_state": prior_weight,
            "var_state": prior_var,
        }

    def update_fn(state_pred, x, kalman_funs):
        wgt_meas, mean_meas, var_meas = interrogate(
            key=x["key"],
            ode_fun=ode_fun,
            ode_weight=ode_weight,
            # t=t_min + (t_max - t_min) * (x["t"] + 1) / n_steps,
            t=x["t"],
            mean_state_pred=state_pred[0],
            var_state_pred=state_pred[1],
        )
        return {
            "x_meas": x_meas,
            "mean_meas": mean_meas,
            "wgt_meas": ode_weight + wgt_meas,
            "var_meas": var_meas,
        }

    stack_fn = None

    # apply filter
    res = utils.kalman_filter(
        state_init=state_init,
        obs=obs,
        predict_fn=predict_fn,
        update_fn=update_fn,
        stack_fn=stack_fn,
        kalman_funs=kalman_funs,
    )

    # append initial values
    res["state_filt"] = (
        utils.append_first(res["state_filt"][0], state_init[0]),
        utils.append_first(res["state_filt"][1], state_init[1]),
    )
    res["state_pred"] = (
        utils.append_first(res["state_pred"][0], state_init[0]),
        utils.append_first(res["state_pred"][1], state_init[1]),
    )
    return res
