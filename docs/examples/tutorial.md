---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Introduction to RODEO: pRobabilistic ODE sOlver

+++

## Description

**rodeo** is a Python library that uses [probabilistic numerics](http://probabilistic-numerics.org/) to solve ordinary differential equations (ODEs).  That is, most ODE solvers (such as [Euler's method](https://en.wikipedia.org/wiki/Euler_method)) produce a deterministic approximation to the ODE on a grid of size $\delta$.  As $\delta$ goes to zero, the approximation converges to the true ODE solution.  Probabilistic solvers also output a solution an a grid of size $\delta$; however, the solution is random.  Still, as $\delta$ goes to zero we get the correct answer.

For the basic task of solving ODEs, **rodeo** provides a probabilistic solver, `rodeo`, for univariate process x(t) of the form

\begin{equation*}
  \WW \xx(t) = f(\xx(t), t), \qquad t \in [a, b], \quad \xx(0) = \vv,
\end{equation*}

where $\xx(t) = \big(x^{(0)}(t), x^{(1)}(t), ..., x^{(q-1)}(t)\big)$ consists of $x(t)$ and its first $q-1$ derivatives, $\WW$ is a coefficient matrix, and $f(\xx(t), t)$ is typically a nonlinear function. T

`rodeo` begins by putting a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) prior on the ODE solution, and updating it sequentially as the solver steps through the grid.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from rodeo.ibm import ibm_init
from rodeo.ode import *
from jax.config import config
config.update("jax_enable_x64", True)
```

## Walkthrough


To illustrate the set-up, let's consider the following ODE example (**FitzHugh-Nagumo** model) where $q=2$ for both variables:

\begin{align*}
\frac{dV}{dt} &= c(V - \frac{V^3}{3} + R), \\
\frac{dR}{dt} &= -\frac{(V - a - bR)}{c}, \\
\xx_0 &= (V(0), R(0)) = (-1,1).
\end{align*}

where the solution $x(t)$ is sought on the interval $t \in [0, 40]$ and $\theta = (a,b,c) = (.2,.2,3)$.  

To approximate the solution with the probabilistic solver, the Gaussian process prior we will use is a so-called [Continuous Autoregressive Process](https://CRAN.R-project.org/package=cts/vignettes/kf.pdf) of order $p$. A particularly simple $\car(p)$ proposed by [Schober et al (2019)](http://link.springer.com/10.1007/s11222-017-9798-7) is the $p-1$ times integrated Brownian motion, 

\begin{equation*}
\xx(t) \sim \ibm(p).
\end{equation*}

Here $\xx(t)  = (x^{(0)}(t), ..., x^{(p-1)}(t))$ consists of $x(t)$ and its first $p-1$ derivatives.
The $\ibm$ model specifies that each of $\xx(t)  = (x^{(0)}(t), ..., x^{(p-1)}(t))$ is continuous, but $x^{(p)}(t)$ is not. Therefore, we need to pick $p > q$. It's usually a good idea to have $p$ a bit larger than $q$, especially when we think that the true solution $x(t)$ is smooth. However, increasing $p$ also increases the computational burden, and doesn't necessarily have to be large for the solver to work.  For this example, we will use $p=3$. To initialize, we simply set $\xx(0) = (\xx_0, 0)$. It is also possible to initialize $\xx(0)$ by computing the higher derivatives but our experiments show that this does not make much of a difference. In this example, there are two variates so $\xx(t)$ is stacked creating a matrix with dimensions $2 \times p$. In a similar fashion, $\WW$ needs to be stacked to create a 3d array of dimension $2 \times 1 \times p$ where $2$ is from the number of variables and $1$ is from the size of the output of $f$ for each variable. The Python code to implement all this is as follows.

```{code-cell} ipython3
def ode_fun_jax(X_t, t, theta):
    "FitzHugh-Nagumo ODE."
    a, b, c = theta
    V, R = X_t[:,0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                    [-1/c*(V - a + b*R)]])


# problem setup and intialization
n_obs = 2  # Total observations
n_deriv_prior = 3 # p

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_steps = 800
tmin = 0.
tmax = 40.
theta = jnp.array([0.2, 0.2, 3])

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = .01
sigma = jnp.array([sigma]*n_obs)

# Initial W for jax block
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

# Initial x0 for jax block
x0_block = jnp.array([[-1., 1., 0.], [1., 1/3, 0.]])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_steps
n_order = jnp.array([n_deriv_prior]*n_obs)
ode_init = ibm_init(dt, n_order, sigma)
```

One of the key steps in the probabilisitc solver is the interrogation step. We offer several choices for this task: `interrogate_schober` by [Schober et al (2019)](http://link.springer.com/10.1007/s11222-017-9798-7), `interrogate_chkrebtii` by [Chkrebtii et al (2016)](https://projecteuclid.org/euclid.ba/1473276259), `interrogate_rodeo` which is a mix of the two, and `interrogate_tronarp` by [Tronarp et al (2018)](http://arxiv.org/abs/1810.03440). 


- `interrogate_schober` is the simplest and fastest.
- `interrogate_chkrebtii` is a Monte Carlo method that returns a non-deterministic output which does not assume zero variance like [Schober et al (2019)](http://link.springer.com/10.1007/s11222-017-9798-7) and [Tronarp et al (2018)](http://arxiv.org/abs/1810.03440).
- `interrogate_rodeo` combines the zeroth order Taylor expansion of [Schober et al (2019)](http://link.springer.com/10.1007/s11222-017-9798-7) and the variance of [Chkrebtii et al (2016)](https://projecteuclid.org/euclid.ba/1473276259).
- `interrogate_tronarp` is an extension to `interrogate_schober` where a first order Taylor approximation is used which has shown to have better numerical stability.

For simple problems such as this one, we recommend `interrogate_rodeo` because it is fast and accurate. For more complex ODEs, we recommend `interrogate_tronarp`. However, the best interrogation method may depend on your specific problem.

+++

`rodeo` offers two output functions: `solve_sim` and `solve_mv`. The former returns a sample from the solution posterior, $\xx_{0:N}$ and the latter returns the posterior mean, $\mmu_{0:N}$ and variance $\SSi_{0:N|N}$. The next step is optional but significantly speeds up the solver if the ODE is solved many times. This uses `jax.jit` to jit-compile the solver. Note that there are 3 static arguments which are either function or integer inputs.

```{code-cell} ipython3
# Jit solver
key = jax.random.PRNGKey(0)
sim_jit = jax.jit(solve_sim, static_argnums=(1, 7, 11))
mv_jit = jax.jit(solve_mv, static_argnums=(1, 7, 11))

xt = sim_jit(key=key, fun=ode_fun_jax,
             W=W_block, x0=x0_block, theta=theta,
             tmin=tmin, tmax=tmax, n_steps=n_steps, **ode_init,
             interrogate=interrogate_rodeo)
mut, _ = mv_jit(key=key, fun=ode_fun_jax,
                W=W_block, x0=x0_block, theta=theta,
                tmin=tmin, tmax=tmax, n_steps=n_steps, **ode_init,
                interrogate=interrogate_rodeo)
```

To compare the `rodeo` solution, we use the deterministic solution provided by `odeint`.

```{code-cell} ipython3
def ode_fun(X_t, t, theta):
    a, b, c = theta
    V, R = X_t
    return np.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

# Initial x0 for odeint
ode0 = np.array([-1., 1.])

# Get odeint solution for Fitz-Hugh
tseq = np.linspace(tmin, tmax, n_steps+1)
exact = odeint(ode_fun, ode0, tseq, args=(theta,))

# Graph the results
fig, axs = plt.subplots(2, 1, figsize=(20, 7))
ylabel = ['V', 'R']
plt.rcParams.update({'font.size': 20})
for i in range(2):
    axs[i].plot(tseq, xt[:, i, 0], label="rodeo sample")
    axs[i].plot(tseq, mut[:, i, 0], label="rodeo mean")
    axs[i].set_ylabel(ylabel[i])
    axs[i].plot(tseq, exact[:, i], label='Exact')
    axs[i].legend(loc='upper left')
```
