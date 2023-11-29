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

# Higher-Ordered ODE

+++

In this notebook, we consider a second-ordered ODE:

\begin{equation*}
x^{(2)}(t) = \sin(2t) − x^{(0)}(t), \qquad \xx_0 = (-1, 0, 1),
\end{equation*}

where the solution $x(t)$ is sought on the interval $t \in [0, 10]$.  In this case, the ODE has an analytic solution,

\begin{equation*}
x(t) = \tfrac 1 3 \big(2\sin(t) - 3\cos(t) - \sin(2t)\big).
\end{equation*}

```{code-cell} ipython3
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import rodeo
import rodeo.prior
import rodeo.interrogate

from jax import config
config.update("jax_enable_x64", True)
```

The setup is almost identical to that of the example in the Introduction to **RODEO** notebook. The major difference is to set `n_deriv=4`$(q=4)$ in this example since we are considering an 2nd order ODE.

```{code-cell} ipython3
def higher_fun(x, t, **params):
    """
    Higher-order ODE of Chkrebtii et al in **rodeo** format.
    Args:
        x: JAX array of shape `(1,4)` corrresponding to
           `X = (x, x^(1), x^(2), x^(3))`.
        t: Scalar time variable.

    Returns:
        JAX array of shape `(1,1)` corresponding to `f(x,t)`.

    """
    return jnp.array([[jnp.sin(2 * t) - x[0, 0]]])


W = jnp.array([[[0., 0., 1., 0.]]])  # LHS matrix of ODE
x0 = jnp.array([[-1., 0., 1., 0.]])  # initial value for the IVP

# Time interval on which a solution is sought.
t_min = 0.
t_max = 10.

# ---  Define the prior process ---------------------------------------

n_vars = 1                        # number of variables in the ODE
n_deriv = 4  # max number of derivatives
sigma = jnp.array([.1] * n_vars)  # IBM process scale factor


# ---  Evaluate the ODE solution --------------------------------------

n_steps = 80                  # number of evaluations steps
dt = (t_max - t_min) / n_steps  # step size

# generate the Kalman parameters corresponding to the prior
prior_Q, prior_R = rodeo.prior.ibm_init(
    dt=dt,
    n_deriv=n_deriv,
    sigma=sigma
)

key = jax.random.PRNGKey(0)  # JAX pseudo-RNG key


# deterministic ODE solver: posterior mean
Xt, _ = rodeo.solve_mv(
    key=key,
    # define ode
    ode_fun=higher_fun,
    ode_weight=W,
    ode_init=x0,
    t_min=t_min,
    t_max=t_max,
    # solver parameters
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_chkrebtii,
    prior_weight=prior_Q,
    prior_var=prior_R
)

# probabilistic ODE solver: draw from posterior
Xt = rodeo.solve_sim(
    key=key,
    # define ode
    ode_fun=higher_fun,
    ode_weight=W,
    ode_init=x0,
    t_min=t_min,
    t_max=t_max,
    # solver parameters
    n_steps=n_steps,
    interrogate=rodeo.interrogate.interrogate_kramer,
    prior_weight=prior_Q,
    prior_var=prior_R
)
```

To see how well this approximation does against the exact solution, we can graph them together. First, we will define the functions of the exact solution for this example.

```{code-cell} ipython3
# Exact Solution for x_t^{(0)}
def ode_exact_x(t):
    return (-3*cos(t) + 2*sin(t) - sin(2*t))/3

# Exact Solution for x_t^{(1)}
def ode_exact_x1(t):
    return (-2*cos(2*t) + 3*sin(t) + 2*cos(t))/3
```

```{code-cell} ipython3
# Get exact solutions for x^{(0)}, x^{(1)}
tseq = np.linspace(t_min, t_max, n_steps+1)
exact_x = np.zeros(n_steps+1)
exact_x1 = np.zeros(n_steps+1)
for t in range(n_steps+1):
    exact_x[t] = ode_exact_x(tseq[t])
    exact_x1[t] = ode_exact_x1(tseq[t])

# Plot them
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
axs[0].plot(tseq, Xt[:,0,0], label = 'rodeo')
axs[0].plot(tseq, exact_x, label = 'Exact')
axs[0].set_title("$x^{(0)}_t$")
axs[0].legend(loc='upper left')
axs[1].plot(tseq, Xt[:,0,1], label = 'rodeo')
axs[1].plot(tseq, exact_x1, label = 'Exact')
axs[1].set_title("$x^{(1)}_t$")
plt.show()
```
