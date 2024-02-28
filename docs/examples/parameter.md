---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Parameter Inference
In this notebook, we demonstrate the steps to conduct parameter inference using various likelihood approximation methods in **rodeo**.

```{code-cell} ipython3
# --- 0. Import libraries and modules ------------------------------------


import jax
import jax.numpy as jnp
import jaxopt
import blackjax
import rodeo
import rodeo.interrogate
import rodeo.inference
import rodeo.prior


import matplotlib.pyplot as plt
import seaborn as sns
from jax import config
config.update("jax_enable_x64", True)
```

## Walkthrough

We will use the FitzHugh-Nagumo model as the example here which is a two-state ODE on $\xx(t) = (V(t), R(t))$,

\begin{align*}
\frac{dV(t)}{dt} &= c(V(t) - \frac{V(t)^3}{3} + R(t)), \\
\frac{dR(t)}{dt} &= -\frac{(V(t) - a - bR(t))}{c}. \\
\end{align*}

The model parameters are $\tth = (a,b,c,V(0),R(0))$, with $a,b,c > 0$ which are to be learned from the measurement model

\begin{align*}
\YY_i \sim \N(\xx(t_i), \phi^2 \Id_{2 \times 2})
\end{align*}

where $t_i = i$ and $i=0,1,\ldots 40$ and $\phi^2 = .04$. We will first simulate some noisy data using an highly accurate ODE solver (`odeint`).

```{code-cell} ipython3
# --- 1. Define the ODE-IVP ----------------------------------------------


def fitz_fun(X, t, **params):
    "FitzHugh-Nagumo ODE in rodeo format."
    a, b, c = params["theta"]
    V, R = X[:, 0]
    return jnp.array(
        [[c * (V - V * V * V / 3 + R)],
         [-1 / c * (V - a + b * R)]]
    )


def fitz_init(x0, theta):
    "FitzHugh-Nagumo initial values in rodeo format."
    x0 = x0[:, None]
    return jnp.hstack([
        x0,
        fitz_fun(X=x0, t=0., theta=theta),
        jnp.zeros_like(x0)
    ])


W = jnp.array([[[0., 1., 0.]], [[0., 1., 0.]]])  # LHS matrix of ODE
x0 = jnp.array([-1., 1.])  # initial value for the ODE-IVP
theta = jnp.array([.2, .2, 3])  # ODE parameters
X0 = fitz_init(x0, theta)  # initial value in rodeo format

# Time interval on which a solution is sought.
t_min = 0.
t_max = 40.

# --- Define the prior process -------------------------------------------

n_vars = 2
n_deriv = 3

# IBM process scale factor
sigma = jnp.array([.1] * n_vars)


# --- data simulation ------------------------------------------------------


dt_obs = 1.  # interobservation time
n_steps_obs = int((t_max - t_min) / dt_obs)
# observation times
obs_times = jnp.linspace(t_min, t_max,
                         num=n_steps_obs + 1)

# number of simulation steps per observation
n_res = 20
n_steps = n_steps_obs * n_res
# simulation times
sim_times = jnp.linspace(t_min, t_max,
                         num=n_steps + 1)

# prior parameters
dt_sim = (t_max - t_min) / n_steps  # grid size for simulation
prior_Q, prior_R = rodeo.prior.ibm_init(
    dt=dt_sim,
    n_deriv=n_deriv,
    sigma=sigma
)

# Produce a Pseudo-RNG key
key = jax.random.PRNGKey(0)


# calculate ODE via deterministic output
key, subkey = jax.random.split(key)
Xt, _ = rodeo.solve_mv(
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
    prior_weight=prior_Q,
    prior_var=prior_R
)

# generate observations
noise_sd = 0.2  # Standard deviation in noise model
key, subkey = jax.random.split(key)
eps = jax.random.normal(
    key=subkey,
    shape=(obs_times.size, 2)
)
# 0th order derivatives at observed timepoints
obs_ind = jnp.searchsorted(sim_times, obs_times)
x = Xt[obs_ind, :, 0]
Y = x + noise_sd * eps
```

Observations are available at $t=0,1,\ldots, 40$. However, the ODE solver requires a higher resolution than $\Delta t = 1$ to give a good approximation. Here we use `n_res=20` which means $\Delta t = 1/20$.

```{code-cell} ipython3
# plot one graph
plt.rcParams.update({'font.size': 30})
fig1, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].plot(obs_times, x[:,0], label = 'True', linewidth=4)
axs[0].scatter(obs_times, Y[:,0], label = 'Obs', color='orange', s=200, zorder=2)
axs[0].set_title("$V(t)$")
axs[1].plot(obs_times, x[:,1], label = 'True', linewidth=4)
axs[1].scatter(obs_times, Y[:,1], label = 'Obs', color='orange', s=200, zorder=2)
axs[1].set_title("$R(t)$")
axs[1].legend(loc=1)
fig1.tight_layout()
```

We proceed with a Bayesian approach by postulating a prior distribution $\pi(\tth)$ which combined with the likelihood gives the posterior
\begin{equation}\label{eq:likelihood}
    p(\tth \mid \YY_{0:M}) \propto \pi(\tth) \times p(\YY_{0:M} \mid \ZZ_{0:N} = \bz, \tth)
\end{equation}
where $p(\YY_{0:M} \mid \ZZ_{0:N} = \bz, \tth)$ is approximated with different methods.
Parameter inference is then accomplished by way of a Laplace approximation, for which we have
\begin{equation*}
    \tth \mid \YY_{0:M} \approx \N(\hat \tth, \hat \VV_{\tth}),
\end{equation*}
where $\hat \tth = \argmax_{\tth} \log p(\tth \mid \YY_{0:M})$ and $\hat \VV_{\tth} = -\big[\frac{\partial^2}{\partial \tth \partial \tth'} \log p(\hat \tth \mid \YY_{0:M})\big]^{-1}$. For the prior, we assume independent $\N(0, 10^2)$ priors on $\log a, \log b, \log c$ and $V(0), R(0)$. The likelihood is defined according to the measurement model above. Here we define the helper function `constrain_pars` to help initialize the process prior and the initial value needed for the underlying ODE solver. Finally `fitz_laplace` draws samples via the Laplace approximation.

```{code-cell} ipython3
# --- parameter inference: basic + laplace -------------------------------

def fitz_logprior(upars):
    "Logprior on unconstrained model parameters."
    n_theta = 5  # number of ODE + IV parameters
    lpi = jax.scipy.stats.norm.logpdf(
        x=upars[:n_theta],
        loc=0.,
        scale=10.
    )
    return jnp.sum(lpi)


def fitz_loglik(obs_data, ode_data, **params):
    """
    Loglikelihood for measurement model.

    Args:
        obs_data (ndarray(n_obs, n_vars)): Observations data.
        ode_data (ndarray(n_obs, n_vars, n_deriv)): ODE solution.
    """
    ll = jax.scipy.stats.norm.logpdf(
        x=obs_data,
        loc=ode_data[:, :, 0],
        scale=noise_sd
    )
    return jnp.sum(ll)


def constrain_pars(upars, dt):
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
    X0 = fitz_init(x0, theta)
    sigma = upars[5:]
    Q, R = rodeo.prior.ibm_init(
        dt=dt,
        n_deriv=n_deriv,
        sigma=sigma
    )
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
        jit=True
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
        key=key,
        mean=uode_mean,
        cov=uode_var,
        shape=(n_samples,)
    )
    # convert back to original scale
    ode_sample = uode_sample.at[:, :3].set(jnp.exp(uode_sample[:, :3]))
    return ode_sample
```

### Basic Likelihood

To start, we use the `basic` method to construct a likelihood approximation. In this method, we approximate $p(\YY_{0:M} \mid \ZZ_{0:N} = \bz, \tth)$ in (1) with $\prod_{i=0}^M p(\YY_i \mid \XX(t_i) = \mmu_{n(i)|N}, \tth)$ where $n(i)$ maps the corresponding time points of the solver to the data and $\mmu_{0:N|N}$ is the ODE posterior mean from the probabilistic solver `solve`. The inputs for `basic` is very similar to the ODE solver `solve`. Additional arguments include `obs_data` which is the observations, `obs_times` which are the time points where data is observed and `obs_loglik` which is the measurement likelihood function.

```{code-cell} ipython3
def neglogpost_basic(upars):
    "Negative logposterior for basic approximation."
    # solve ODE
    theta, X0, prior_Q, prior_R = constrain_pars(upars, dt_sim)
    # basic loglikelihood
    ll = rodeo.inference.basic(
        key=key,  # immaterial, since not used
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
        obs_loglik=fitz_loglik
    )
    return -(ll + fitz_logprior(upars))

# optimization process
n_samples = 100000
upars_init = jnp.append(jnp.log(theta), x0)
upars_init = jnp.append(upars_init, jnp.ones(n_vars))
basic_post = fitz_laplace(key, neglogpost_basic, n_samples, upars_init)
```

### Chkrebtii MCMC

Another method to estimate the parameter posteriors of $\tth$ is given by [Chkrebtii et al (2016)](https://projecteuclid.org/euclid.ba/1473276259) using MCMC. First, $\tth_0$ is initialized from a given prior $\pi(\tth)$. Next, a sample solution, $\xx_{0:N}$ dependent on $\tth_0$ is computed from `solve`. At each sampling step, $\tth' \sim q(\tth_{i-1})$ is sampled from the proposal distribution and is used to compute a new sample solution, $\xx_{0:N}'$. Finally, a rejection ratio is used to decide if $\tth_i = \tth'$ is updated or $\tth_i = \tth_{i-1}$ is kept. 

The structure of this method is different than the other solvers presented here. That is, the MCMC method uses a base class to implement the skeleton of the algorithm with a few functions that need to be implemented by the user. First the `logprior` and `obs_loglik` methods define the log-prior and loglikelihood respectively. Next, the `parse_pars` method is analogous to the `constrain_pars` function which helps with the initialization of the process prior and the initial values for the ODE solver. 

```{code-cell} ipython3
class fitz_mcmc(rodeo.inference.MarginalMCMC):

    def logprior(self, upars):
        "Need to implement this to define the prior"
        return fitz_logprior(upars)

    def obs_loglik(self, obs_data, ode_data, **params):
        "Need to implement this to define the observation likelihood"
        return fitz_loglik(obs_data, ode_data, **params)

    def parse_pars(self, upars, dt):
        "Need to implement this to parse the parameters"
        theta, X0, prior_Q, prior_R, = constrain_pars(upars, dt)
        params ={
            "theta": theta
        }
        return W, X0, prior_Q, prior_R, params

    # you can define your solve function like this or use the default
    def solve(self, key, ode_weight, ode_init, prior_weight, prior_var, **params):
        "You can define your solve function like this or use the base class solve."
        Xt = rodeo.solve_sim(
            key=key,
            # define ode
            ode_fun=self._ode_fun,
            ode_weight=ode_weight,
            ode_init=ode_init,
            t_min=self._t_min,
            t_max=self._t_max,
            # solver parameters
            n_steps=self._n_steps,
            interrogate=rodeo.interrogate.interrogate_kramer,
            prior_weight=prior_weight,
            prior_var=prior_var,
            **params
        )
        return Xt


def fitz_chmcmc(key, n_samples, upars_init):
    r"""
    Sample via the Marginal MCMC algorithm.

    Args:
        key : PRNG key.
        n_samples : Number of samples to return.
        upars_init : Initial parameter to be optimized.


    Return:
        JAX array of shape ``(n_samples, 5)`` of posterior
        samples from ``(theta, x0)``.
    """
    key, *subkeys = jax.random.split(key, num=3)
    # proposal parameter
    scale = jnp.array(
        [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

    # initial the fitz_mcmc class
    fitz_ch = fitz_mcmc(
        ode_fun=fitz_fun,
        obs_data=Y,
        obs_times=obs_times,
        t_min=t_min,
        t_max=t_max,
        n_steps=n_steps)

    # compute initial state for initial parameters
    initial_state = fitz_ch.initialize(subkeys[0], upars_init)

    # inference loop similar to blackjax api
    def inference_loop(key, initial_state, n_samples):

        def one_step(state, key):
            state, sample = fitz_ch.step(key, state, scale=scale)
            return state, sample

        keys = jax.jax.random.split(key, n_samples)
        _, samples = jax.lax.scan(one_step, initial_state, keys)
        return samples

    uode_sample = inference_loop(
        subkeys[1], initial_state, n_samples)["Theta"]
    # convert back to original scale
    ode_sample = uode_sample.at[:, :3].set(jnp.exp(uode_sample[:, :3]))
    return ode_sample

# optimization process
n_samples = 10000
upars_init = jnp.append(jnp.log(theta), x0)
upars_init = jnp.append(upars_init, .1*jnp.ones(n_vars))
mcmc_post = fitz_chmcmc(key, n_samples, upars_init)
```

### Fenrir

`fenrir` is a method developed by [Tronarp et al (2022)](https://proceedings.mlr.press/v162/tronarp22a.html) which uses the data itself in the solution process. `fenrir` begins by using the data-free forward pass of `solve` to estimate $p(\XX_{0:N} \mid \ZZ_{0:N} = \bz, \tth)$. This model can be simulated from via a (non-homogeneous) Markov chain going backwards in time,
\begin{equation}\label{eq:fenrirback}
    \begin{aligned}
        \XX_N & \sim \N(\bb_N, \CC_N) \\
        \XX_n \mid \XX_{n+1} & \sim \N(\AA_n \XX_n + \bb_n, \CC_n),
    \end{aligned}
\end{equation}
where the coefficients $\AA_{0:N-1}$, $\bb_{0:N}$, and $\CC_{0:N}$ can be derived using the Kalman filtering and smoothing recursions. Next, `fenrir` assumes that Gaussian observations are added to the model, from which $p(\YY_{0:M} \mid \ZZ_{0:N} = \bz, \tth)$ is computed using a Kalman filter on the backward pass of (2).

To use `fenrir`, we first need to define the specifications. `fenrir` expects observations to be of the form
\begin{equation*}
\YY_i \sim \N(\DD_i \XX_i, \OOm_i).
\end{equation*}
This translates to the following set of definitions for this 2-state ODE.

```{code-cell} ipython3
# gaussian measurement model specification in blocked form
n_meas = 1  # number of measurements per variable in obs_data_i
obs_data = jnp.expand_dims(Y, axis=-1)
obs_weight = jnp.zeros((len(obs_data), n_vars, n_meas, n_deriv))
obs_weight = obs_weight.at[:].set(jnp.array([[[1., 0., 0.]], [[1., 0., 0.]]]))
obs_var = jnp.zeros((len(obs_data), n_vars, n_meas, n_meas))
obs_var = obs_var.at[:].set(noise_sd**2 * jnp.array([[[1.]], [[1.]]]))
```

The way to construct a likelihood estimation is very similar to the basic method. The one difference is that `fenrir` asks for `obs_weight` and `obs_var` instead of `obs_loglik` since it assumes observations are multivariate normal.

```{code-cell} ipython3
def neglogpost_fenrir(upars):
    "Negative logposterior for basic approximation."
    theta, X0, prior_Q, prior_R = constrain_pars(upars, dt_sim)
    # fenrir loglikelihood
    ll = rodeo.inference.fenrir(
        key=key,  # immaterial, since not used
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
        obs_var=obs_var
    )
    return -(ll + fitz_logprior(upars))

# optimization process
n_samples = 100000
upars_init = jnp.append(jnp.log(theta), x0)
upars_init = jnp.append(upars_init, jnp.ones(n_vars))
fenrir_post = fitz_laplace(key, neglogpost_fenrir, n_samples, upars_init)
```

### Dalton

Finally, we present the method, `dalton`, developed by [Wu, Lysy](https://arxiv.org/pdf/2306.05566.pdf). `dalton` uses the model
\begin{equation}
    \begin{aligned}
        \XX_{n+1} \mid \XX_n & \sim \N(\QQ \XX_n, \RR) \\
        \ZZ_n & \sim \N(\WW \XX_n - \ff(\XX_n, t_n, \tth), \VV_n) \\
        \YY_i & \sim p(\YY_i \mid \XX_{n(i)}, \pph),
    \end{aligned}
\end{equation}

where the data is used directly in the forward pass instead of just the backward pass of `fenrir`. For Gaussian observations such as this example, `dalton` is the appropriate function to use.

```{code-cell} ipython3
def neglogpost_dalton(upars):
    "Negative logposterior for basic approximation."
    theta, X0, prior_Q, prior_R = constrain_pars(upars, dt_sim)
    # fenrir loglikelihood
    ll = rodeo.inference.dalton(
        key=key,  # immaterial, since not used
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
        obs_var=obs_var
    )
    return -(ll + fitz_logprior(upars))

# optimization process
n_samples = 100000
upars_init = jnp.append(jnp.log(theta), x0)
upars_init = jnp.append(upars_init, jnp.ones(n_vars))
dalton_post = fitz_laplace(key, neglogpost_dalton, n_samples, upars_init)
```

## Results

We compare the likelihood estimation for the four methods. Only Chrebtii MCMC algorithm differs from the rest because it uses MCMC to sample rather than the Laplace approximation.

```{code-cell} ipython3
plt.rcParams.update({'font.size': 15})
param_true = jnp.append(theta, x0)
var_names = ['a', 'b', 'c', r"$V(0)$", r"$R(0)$"]
fig, axs = plt.subplots(1, 5, figsize=(20,5))
for i in range(5):
    sns.kdeplot(basic_post[:, i], ax=axs[i], label='basic')
    sns.kdeplot(mcmc_post[:, i], ax=axs[i], label='mcmc')
    sns.kdeplot(fenrir_post[:, i], ax=axs[i], label='fenrir')
    sns.kdeplot(dalton_post[:, i], ax=axs[i], label='dalton')
    axs[i].axvline(x=param_true[i], linewidth=1, color='r', linestyle='dashed')
    axs[i].set_ylabel("")
    axs[i].set_title(var_names[i])
axs[0].legend(framealpha=0.5, loc='best')
fig.tight_layout()
```

## Dalton Non-Gaussian Observations

Now suppose that the noisy observation model is
\begin{equation}\label{eq:fitznoiseng}
    Y_{ij} \sim \operatorname{Poisson}(\exp{b_0 + b_1x_j(t_i)}),
\end{equation}
where $b_0 = 0.1$ and $b_1 = 0.5$.

```{code-cell} ipython3
# simulate data
key = jax.random.PRNGKey(100)
b0 = 0.1
b1 = 0.5
Yt = jax.random.poisson(key,lam= jnp.exp(b0+b1*x))
obs_data = jnp.expand_dims(Yt, -1)
```

For non-Gaussian observations, please use `daltonng`. The inputs are the same as `dalton` with `obs_weight` and `obs_var` replaced by `obs_loglik_i`, which is the loglikelihood function of the observation. The rest of the process is the same as the Gaussian example.

```{code-cell} ipython3
def obs_loglik_i(obs_data_i, ode_data_i, ind, **params):
    """
    Likelihood function for the SEIRAH model in non-Gaussian DALTON format.

    Args:
        obs_data_i : Observations at index ``ind``.
        ode_data_i :  ODE solution at index ``ind``.
        ind : Index to determine the observation loglikelihood function.

    Returns:
        Loglikelihood of ``obs_data_i`` at index ``ind``.
    """
    b0 = 0.1
    b1 = 0.5
    return jnp.sum(jax.scipy.stats.poisson.logpmf(obs_data_i.flatten(), jnp.exp(b0+b1*ode_data_i[:, 0])))


def neglogpost_daltonng(upars):
    "Negative logposterior for non-Gaussian DALTON approximation."
    theta, X0, prior_Q, prior_R = constrain_pars(upars, dt_sim)
    # non-Gaussian DALTON loglikelihood
    ll = rodeo.inference.daltonng(
        key=key,  # immaterial, since not used
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
        # non-gaussian measurement model
        obs_data=obs_data,
        obs_times=obs_times,
        obs_loglik_i=obs_loglik_i
    )
    return -(ll + fitz_logprior(upars))

# optimization process
n_samples = 100000
upars_init = jnp.append(jnp.log(theta), x0)
upars_init = jnp.append(upars_init, jnp.ones(n_vars))
dalton_post = fitz_laplace(key, neglogpost_daltonng, n_samples, upars_init)
```

## Results

```{code-cell} ipython3
fig, axs = plt.subplots(1, 5, figsize=(20,5))
for i in range(5):
    tmp_data = dalton_post[:, i]
    if i==0 or i==1:
        tmp_data = tmp_data[(tmp_data>0) & (tmp_data<3)]
    sns.kdeplot(tmp_data, ax=axs[i], label='dalton')
    axs[i].axvline(x=param_true[i], linewidth=1, color='r', linestyle='dashed')
    axs[i].set_ylabel("")
    axs[i].set_title(var_names[i])
fig.tight_layout()
```
