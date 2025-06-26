# **rodeo:** Probabilistic Methods of Parameter Inference for Ordinary Differential Equations

[**Home**](https://rodeo.readthedocs.io/)
| [**Installation**](#installation)
| [**Documentation**](#documentation)
| [**Tutorial**](#walkthrough)
| [**Developers**](#developers)

---
[![Continuous integration](https://github.com/mlysy/rodeo/actions/workflows/build_test.yml/badge.svg)](https://github.com/mlysy/rodeo/actions/workflows/build_test.yml)

## Description

**rodeo** is a fast Python library that uses [probabilistic numerics](http://probabilistic-numerics.org/) to solve ordinary differential equations (ODEs).  That is, most ODE solvers (such as [Euler's method](https://en.wikipedia.org/wiki/Euler_method)) produce a deterministic approximation to the ODE on a grid of step size $\Delta t$.  As $\Delta t$ goes to zero, the approximation converges to the true ODE solution.  Probabilistic solvers also output a solution on a grid of size $\Delta t$; however, the solution is random.  Still, as $\Delta t$ goes to zero, the probabilistic numerical approximation converges to the true solution. 

**rodeo** provides a lightweight and extensible family of approximations to a nonlinear Bayesian filtering paradigm common to many probabilistic solvers ([Tronarp et al (2018)](http://arxiv.org/abs/1810.03440)). This begins by putting a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) prior on the ODE solution, and updating it sequentially as the solver steps through the grid. **rodeo** is built on **jax** which allows for just-in-time compilation and auto-differentiation. The API of **jax** is almost equivalent to that of **numpy**. 

**rodeo** provides two main tools: one for approximating the ODE solution and the other for parameter inference. For the former we provide:

- `solve`: Implementation of a probabilistic ODE solver which uses a nonlinear Bayesian filtering paradigm.

For the latter we provide the likelihood approximation methods:

- `basic`: Implementation of a basic likelihood approximation method (details can be found in [Wu and Lysy (2024)](https://proceedings.mlr.press/v238/wu24b.html)).
- `fenrir`: Implementation of Fenrir ([Tronarp et al (2022)](https://proceedings.mlr.press/v162/tronarp22a.html)).
- `random_walk_aux`: MCMC implementation of Chkrebtii's method ([Chkrebtii et al (2016)](https://projecteuclid.org/euclid.ba/1473276259)).
- `dalton`: Implementation of our data-adaptive ODE likelihood approximation ([Wu and Lysy (2024)](https://proceedings.mlr.press/v238/wu24b.html)).
- `magi`: Implementation of MAGI ([Wong et al (2023)](https://arxiv.org/abs/2203.06066)) with a Markov prior.

Detailed examples for their usage can be found in the [Documentation](#documentation) section. Please note that this is the **jax**-only version of **rodeo**. For the legacy versions using various other backends please see [here](https://github.com/mlysy/rodeo-legacy).

## Installation

Download the repo from GitHub and then install with the `setup.cfg` script:
```bash
git clone https://github.com/mlysy/rodeo.git
cd rodeo
pip install .
```

## Documentation

Please first go to [readthedocs](https://rodeo.readthedocs.io/) to see the rendered documentation for the following examples. 

- A [quickstart tutorial](docs/examples/tutorial.md) on solving a simple ODE problem.

- An example for solving a [higher-ordered ODE](docs/examples/higher_order.md).

- An example for solving a difficult [chaotic ODE](docs/examples/lorenz.md).

- An example of a parameter inference problem where we use the [Laplace approximation](docs/examples/parameter.md).

## Walkthrough

In this walkthrough, we show both how to solve an ODE with our probabilistic solver and conduct parameter inference. 
We will first illustrate the set-up for solving the ODE. To that end, let's consider the following first ordered ODE example (**FitzHugh-Nagumo** model),

$$
\begin{align*}
    \frac{dV}{dt} &= c(V - \frac{V^3}{3} + R), \\
    \frac{dR}{dt} &= -\frac{(V - a + bR)}{c}, \\
    X(t) &= (V(0), R(0)) = (-1,1).
\end{align*}
$$

where the solution $X(t)$ is sought on the interval $t \in [0, 40]$ and $\theta = (a,b,c) = (.2,.2,3)$. 

Following the notation of [Wu and Lysy (2024)](https://proceedings.mlr.press/v238/wu24b.html), we have $p-1=1$ in this example. To approximate the solution with the probabilistic solver, 
we use a simple Gaussian process prior proposed by [Schober et al (2019)](http://link.springer.com/10.1007/s11222-017-9798-7); namely, that $V(t)$ and $R(t)$ are 
independent $q-1$ times integrated Brownian motion, such that 

$$
\begin{equation*}
x^{(q)}(t) = \sigma_x B(t)
\end{equation*}
$$

for $x=V, R$. The result is a $q$-dimensional continuous Gaussian Markov process $\boldsymbol{x(t)} = \big(x^{(0)}(t), x^{(1)}(t), \ldots, x^{(q-1)}(t)\big)$
for each variable $x=V, R$. Here $x^{(i)}(t)$ denotes the $i$-th derivative of $x(t)$. The IBM model specifies that each of these is continuous, but $x^{(q)}(t)$ is not. 
Therefore, we need to pick $q \geq p$. It's usually a good idea to have $q$ a bit larger than $p$, especially when 
we think that the true solution $X(t)$ is smooth. However, increasing $q$ also increases the computational burden, 
and doesn't necessarily have to be large for the solver to work.  For this example, we will use $q=3$. 
To initialize, we simply set $\boldsymbol{X(0)} = (V^{(0)}(0), V^{(1)}(0), 0, R^{(0)}(0), R^{(1)}(0), 0)$ where we padded the initial value with zeros for the higher derivative. 
The Python code to implement all this is as follows.

```python
import jax
import jax.numpy as jnp
import rodeo

def fitz_fun(X, t, **params):
    "FitzHugh-Nagumo ODE in rodeo format."
    a, b, c = params["theta"]
    V, R = X[:, 0]
    return jnp.array(
        [[c * (V - V * V * V / 3 + R)],
         [-1 / c * (V - a + b * R)]]
    )

n_vars = 2  # number of variables in the ODE
n_deriv = 3  # max number of derivatives

x0 = jnp.array([-1., 1.])  # initial value for the ODE-IVP
theta = jnp.array([.2, .2, 3])  # ODE parameters

# we have a helper function to help with the rodeo initialization
W, fitz_init_pad = rodeo.utils.first_order_pad(fitz_fun, n_vars, n_deriv)
# fitz_init_pad takes Args:
# x0: initial value for the ODE-ivp
# t: initial time
# **params: extra model parameters as kwargs
X0 = fitz_init_pad(x0, 0., theta=theta)  # initial value in rodeo format

# Time interval on which a solution is sought.
t_min = 0.
t_max = 40.

# --- Define the prior process -------------------------------------------

sigma = jnp.array([.1] * n_vars)  # IBM process scale factor


# --- data simulation ------------------------------------------------------

n_steps = 800  # number of evaluations steps
dt = (t_max - t_min) / n_steps  # step size

# generate the Kalman parameters corresponding to the prior
prior_Q, prior_R = rodeo.prior.ibm_init(
    dt=dt_sim,
    n_deriv=n_deriv,
    sigma=sigma
)

# Produce a Pseudo-RNG key
key = jax.random.PRNGKey(0)

Xt, _ = rodeo.solve_mv(
    key=key,
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
```

We compare the solution from the solver to the deterministic solution provided by `odeint` in the **scipy** library. 

![fitzsol](https://raw.githubusercontent.com/mlysy/rodeo/main/docs/figures/fitzsol.png)


We also include examples for solving a [higher-ordered ODE](docs/examples/higher_order.md) and a [chaotic ODE](docs/examples/lorenz.md).

## Parameter Inference

We now move to the parameter inference problem. **rodeo** contains several likelihood approximation methods summarized in the [Description](#description) section.
Here, we will use the `basic` likelihood approximation method. Suppose observations are simulated via the model

$$
Y(t) \sim \textnormal{Normal}(X(t), \phi^2 \cdot \boldsymbol{I}_{2\times 2})
$$

where $t=0, 1, \ldots, 40$ and $\phi^2 = 0.005$. The parameters of interest are $\boldsymbol{\Theta} = (a, b, c, V(0), R(0))$ with $a,b,c > 0$.
We use a normal prior for $(\log a, \log b, \log c, V(0), R(0))$ with mean $0$ and standard deivation $10$. 
The following function can be used to construct the `basic` likelihood approximation for $\boldsymbol{\Theta}$.

```python
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
        scale=0.005
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
    X0 = fitz_init(x0, 0, theta=theta)
    sigma = upars[5:]
    Q, R = rodeo.prior.ibm_init(
        dt=dt,
        n_deriv=n_deriv,
        sigma=sigma
    )
    return theta, X0, Q, R


def neglogpost_basic(upars):
    "Negative logposterior for basic approximation."
    # solve ODE
    theta, X0, prior_Q, prior_R = constrain_pars(upars, dt_sim)
    # basic loglikelihood
    ll = rodeo.inference.basic(
        key=key, 
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
        obs_data=obs_data,
        obs_times=obs_times,
        obs_loglik=fitz_loglik
    )
    return -(ll + fitz_logprior(upars))
```

This is a basic example to demonstrate usage. We suggest more sophisticated likelihood approximations which propagate the solution uncertainty to the likelihood approximation such as `fenrir`, `marginal_mcmc` and `dalton`. Please refer to the [parameter inference tutorial](docs/examples/parameter.md) for more details.

## Results

Here are some results produced by various likelihood approximations found in **rodeo** from `/examples/`:

### FitzHugh-Nagumo

![fitzhugh](https://raw.githubusercontent.com/mlysy/rodeo/main/docs/figures/fitzfigure.png)

### Hes1

![hes1](https://raw.githubusercontent.com/mlysy/rodeo/main/docs/figures/hes1figure.png)

### SEIRAH

![seirah](https://raw.githubusercontent.com/mlysy/rodeo/main/docs/figures/seirahfigure.png)

## Developers

### Unit Testing

The unit tests can be ran through the following commands:
```bash
cd tests
python -m unittest discover -v
```

Or, install [**tox**](https://tox.wiki/en/latest/index.html), then from within `rodeo` enter command line: `tox`.

### Building Documentation

The HTML documentation can be compiled from the root folder:
```bash
pip install .[docs]
cd docs
make html
```
This will create the documentation in `docs/build`.
