r"""
Computes the initial parameters for the process prior using the p-1 times integrated Brownian motion (IBM)

.. math::

    x^{(p-1)}(t) = \sigma B(t).

Note that the IBM process is a particularly simple continuous autoregressive process of the form

.. math::

    x^{(p-1)}(t) + \alpha_1 x^{(p-2)}(t) + \ldots + \alpha_{p-1} x(t) = \sigma B(t),

where :math:`\alpha_1 = \ldots = \alpha_{p-1} = 0`. It has the analytical formulas for the parameters

.. math::

    Q_{ij} = 𝟙_{i\leq j}\frac{(\Delta t)^{j-i}}{(j-1)!}, \qquad R_{ij} = \sigma^2 \frac{(\Delta t)^{2p+1-i-j}}{(2p+1-i-j)(p-i)!(p-j)!}.

"""


import jax
import jax.numpy as jnp
import jax.scipy as jsp

def _factorial(x):
    """
    JAX factorial function.

    It's actually the gamma function shifted such that `_factorial(x) = x!` for integer-valued `x`.

    Args:
        x (int): Integer.
    
    Returns:
        (int): Factorial of x.

    """
    return jnp.exp(jsp.special.gammaln(x+1.0))


def ibm_state(dt, q, sigma):
    """
    Calculate the state weight matrix and variance matrix of q-times integrated Brownian motion.

    The q-times integrated Brownian motion process :math:`X_t` is such that its (q+1)-th order derivative :math:`X^{q+1}_t = d^q/dt^(q+1) X_t` is :math:`\sigma B_t`, i.e., Brownian motion scaled by :math:`\sigma`.

    Args:
        dt (float): The step size between simulation points.
        q (int): The number of times to integrate the underlying Brownian motion.
        sigma (float): Parameter in the q-times integrated Brownian Motion.

    Returns:
        (tuple):
        - **Q** (ndarray(q+1, q+1)): The state weight matrix defined in
          Kalman solver.
        - **R** (ndarray(q+1, q+1)): The state variance matrix defined in
          Kalman solver.

    """
    I, J = jnp.meshgrid(jnp.arange(q+1), jnp.arange(q+1),
                        indexing="ij", sparse=True)
    mesh = J-I
    Q = jnp.nan_to_num(dt**mesh/_factorial(mesh), 0)
    mesh = (2.0*q+1.0) - I - J
    num = dt**mesh
    den = mesh * _factorial(q - I) * _factorial(q-J)
    R = sigma**2 * num/den
    return Q, R


def ibm_init(dt, n_deriv, sigma):
    """
    Calculates the initial parameters necessary for the Kalman solver with the p-1 times
    integrated Brownian Motion.

    Args:
        dt (float): The step size between simulation points.
        n_deriv (int): Dimension of the prior.
        sigma (ndarray(n_block)): Parameter in variance matrix.

    Returns:
        (tuple):
        - **wgt_state** (ndarray(n_block, p, p)) Weight matrix defining the solution prior; :math:`Q`.
        - **var_state** (ndarray(n_block, p, p)) Variance matrix defining the solution prior; :math:`R`.

    """
    n_block = len(sigma)

    wgt_state_i, var_state_i = ibm_state(dt, n_deriv-1, 1)
    wgt_state = jnp.repeat(wgt_state_i[None,:,:], repeats=n_block, axis=0)
    var_state = jax.vmap(lambda b:
        sigma[b]**2 * var_state_i)(jnp.arange(n_block))
    
    return wgt_state, var_state
