r"""
Util functions for rodeo.

"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp


def add_sqrt(sqrt_A,sqrt_B):
    r"""
    Transforms the square roots of matrices A and B into the square root of their sum.

    Args:
        sqrt_A (ndarray(n_dim, n_dim)): The square root of matrix A.
        sqrt_B (ndarray(n_dim, n_dim)): The square root of matrix B.

    Returns:
        (ndarray(n_dim, n_dim)): The square root of the sum of matrices A and B.
    """

    sqrt_sum = jnp.vstack([sqrt_A.T,sqrt_B.T])
    Q,R = jnp.linalg.qr(sqrt_sum)
    return R.T


def solve_var(V, B):
    r"""
    Computes :math:`X = V^{-1}B`, where :math:`V` is a variance matrix.

    Args:
        V (ndarray(n_dim1, n_dim1)): Variance matrix :math:`V`.
        B (ndarray(n_dim1, n_dim2)): Matrix :math:`B`.

    Returns:
        (ndarray(n_dim1, n_dim2)): Matrix :math:`X = V^{-1}B`.
    """

    # L, low = jsp.linalg.cho_factor(V)
    # return jsp.linalg.cho_solve((L, low), B)
    return jnp.linalg.solve(V, B)


def mvncond(mu, Sigma, icond):
    """
    Calculates A, b, and V such that :math:`y[!icond] | y[icond] \sim \operatorname{Normal}(A y[icond] + b, V)`.

    Args:
        mu (ndarray(2*n_dim)): Mean of y.
        Sigma (ndarray(2*n_dim, 2*n_dim)): Covariance of y. 
        icond (ndarray(2*nd_dim)): Conditioning on the terms given.

    Returns:
        (tuple):
        - **A** (ndarray(n_dim, n_dim)): For :math:`y \sim \operatorname{Normal}(\mu, \Sigma)` 
          such that :math:`y[!icond] | y[icond] \sim \operatorname{Normal}(A y[icond] + b, V)` Calculate A.
        - **b** (ndarray(n_dim)): For :math:`y \sim \operatorname{Normal}(\mu, \Sigma)` 
          such that :math:`y[!icond] | y[icond] \sim \operatorname{Normal}(A y[icond] + b, V)` Calculate b.
        - **V** (ndarray(n_dim, n_dim)): For :math:`y \sim \operatorname{Normal}(\mu, \Sigma)`
          such that :math:`y[!icond] | y[icond] \sim \operatorname{Normal}(A y[icond] + b, V)` Calculate V.

    """
    # if y1 = y[~icond] and y2 = y[icond], should have A = Sigma12 * Sigma22^{-1}
    ficond = jnp.nonzero(~icond)
    ticond = jnp.nonzero(icond)
    # A = jnp.dot(Sigma[jnp.ix_(ficond[0], ticond[0])], jsp.cho_solve(
    #     jsp.cho_factor(Sigma[jnp.ix_(ticond[0], ticond[0])]), jnp.identity(sum(icond))))
    A = jnp.dot(Sigma[jnp.ix_(ficond[0], ticond[0])],
                solve_var(Sigma[jnp.ix_(ticond[0], ticond[0])],
                          jnp.identity(jnp.sum(icond)))
                )
    b = mu[~icond] - jnp.dot(A, mu[icond])  # mu1 - A * mu2
    V = Sigma[jnp.ix_(ficond[0], ficond[0])] - jnp.dot(A, Sigma[jnp.ix_(ticond[0], ficond[0])])  # Sigma11 - A * Sigma21
    return A, b, V

def multivariate_normal_logpdf(x, mean, cov):
    r"""Using eigendecomposition to compute multivariate normal logpdf.
    
    Args:
        x (ndarray(p)): Observations.
        mean (ndarray(p)): Mean of the distribution.
        cov (ndarray(p, p)): Symmetric positive (semi)definite covariance matrix of the distribution.
    
    Returns:
        (float): The logpdf of the multivariate normal.
    """
    w, v = jnp.linalg.eigh(cov)
    z = jnp.dot(v.T, x - mean)
    z2 = z**2
    iw = ~jnp.isclose(w, 0, rtol=1e-300)
    w = jnp.where(iw, w, 1.) # remove possibility of nan
    val = z2/w + jnp.log(w)
    val = -.5 * jnp.sum(jnp.where(iw, val, 0.)) - jnp.sum(iw)*.5*jnp.log(2*jnp.pi) 
    return val
