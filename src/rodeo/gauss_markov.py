r"""
Direct calculation of mean and variance for Gaussian Markov models.

Let :math:`Y = (Y_0, \ldots, Y_N)` be p-dimensional vectors which follow a Gaussian Markov process:

.. math::

    Y_0 = b_0 + C_0 \epsilon_0

    Y_n = b_n + A_n Y_{n-1} + C_n \epsilon_n,

where :math:`\epsilon_n` are independent vectors of p-dimensional iid standard normals. 

If they are in the form of a Gaussian state-space model:

.. math::

    x_0 = c_0 + R_0^{1/2} \epsilon_0

    x_n = c_n + Q_n x_{n-1} + R_n^{1/2} \epsilon_n

    y_n = d_n + W_n x_n + V_n^{1/2} \eta_n

then first use `kalman2gm` to convert it to the form of a Gaussian Markov process.

"""
import jax
import jax.numpy as jnp

def gauss_markov_mv(A, b, C):
    r"""
    Direct calculation of mean and variance for Gaussian Markov models.

    Args:
        A (ndarray(n_steps, n_dim, n_dim)): Transition matrices in the Gaussian process, i.e., :math:`(A_0, \ldots, A_N)`.
        b (ndarray(n_steps+1, n_dim)): Transition offsets in the Gaussian process, i.e., :math:`(b_0, \ldots, b_N)`.
        C (ndarray(n_steps+1, n_dim, n_dim)): Cholesky factors of the variance matrices in the Gaussian process, i.e., :math:`(C_0, \ldots, C_N)`.

    Returns:
        (tuple):
        - **mean** (ndarray(n_steps+1, n_dim)): Mean of :math:`Y`.
        - **var** (ndarray(n_steps+1, n_dim, n_steps+1, n_dim)): Variance of :math:`Y`.

    """
    n_tot, n_dim = b.shape  # n_tot = n_steps + 1
    AA = jnp.zeros((n_tot, n_tot, n_dim, n_dim))
    for m in range(n_tot):
        # m = column index
        for n in range(n_tot):
            # n = row index
            if n <= m:
                AA = AA.at[n, m].set(jnp.eye(n_dim))
            else:
                AA = AA.at[n, m].set(A[n-1].dot(AA[n-1, m]))
    # Now we can calculate L and u
    L = jnp.zeros((n_tot, n_dim, n_tot, n_dim))
    for m in range(n_tot):
        # m = column index
        for n in range(m, n_tot):
            # n = row index
            L = L.at[n, :, m, :].set(AA[n, m].dot(C[m]))
    u = jnp.zeros((n_tot, n_dim))
    for n in range(n_tot):
        for m in range(n+1):
            u = u.at[n].set(u[n] + AA[n, m].dot(b[m]))
    # compute V = LL'
    # to do this need to reshape L
    L = jnp.reshape(L, (n_tot*n_dim, n_tot*n_dim))
    V = jnp.reshape(L.dot(L.T), (n_tot, n_dim, n_tot, n_dim))
    return u, V


def kalman2gm(wgt_state, mu_state, var_state, wgt_meas, mu_meas, var_meas):
    r"""
    Converts the parameters of the Gaussian state-space model to the parameters of the Gaussian Markov model.

    Args:
        wgt_state (ndarray(n_steps, n_state, n_state)): Transition matricesin the state model; denoted by :math:`Q_1, \ldots, Q_N`.
        mu_state (ndarray(n_steps+1, n_state)): Offsets in the state model; denoted by :math:`c_0, \ldots, c_N`.
        var_state (ndarray(n_steps+1, n_state, n_state)): Variance matrices in the state model; denoted by :math:`R_0, \ldots, R_N`.
        wgt_meas (ndarray(n_steps, n_meas, n_state)): Transition matrices in the measurement model; denoted by :math:`W_0, \ldots, W_N`.
        mu_meas (ndarray(n_steps+1, n_meas)): Offsets in the measurement model; denoted by :math:`d_0, \ldots, d_N`.
        var_meas (ndarray(n_steps+1, n_meas, n_meas)): Variance matrices in the measurement model; denoted by :math:`V_0, \ldots, V_N`.

    Returns:
        (tuple):
        - **wgt_gm** (ndarray(n_steps, n_dim, n_dim)): Transition matrices in the Gaussian Markov model, where `n_dim = n_state + n_meas`; denoted by :math:`A_1, \ldots, A_N`.
        - **mu_gm** (ndarray(n_steps+1, n_dim)): Offsets in the Gaussian Markov model; denoted by :math:`b_0, \ldots, b_N`.
        - **chol_gm** (ndarray(n_steps+1, n_dim, n_dim)): Cholesky factors of the variance matrices in the Gaussian Markov model; denoted by :math:`C_0, \ldots, C_N`.

    """
    # dimensions
    n_tot, n_meas, n_state = wgt_meas.shape  # n_tot = n_steps + 1
    n_dim = n_state + n_meas
    # increase dimension of wgt_state to simplify indexing
    wgt_state = jnp.concatenate([jnp.zeros((n_state, n_state))[None],
                                wgt_state])
    # useful zero matrices
    zero_sm = jnp.zeros((n_state, n_meas))
    zero_mm = jnp.zeros((n_meas, n_meas))
    # initialize outputs
    mu_gm = jnp.zeros((n_tot, n_dim))
    chol_gm = jnp.zeros((n_tot, n_dim, n_dim))
    # increase dimension of wgt_gm to simplify indexing
    wgt_gm = jnp.zeros((n_tot, n_dim, n_dim))
    for i in range(n_tot):
        # mean term
        mu_gm = mu_gm.at[i].set(jnp.concatenate(
            [mu_state[i],
             mu_meas[i] + wgt_meas[i].dot(mu_state[i])]
        ))
        # weight term
        if i > 0:
            wgt_gm = wgt_gm.at[i].set(jnp.block(
                [[wgt_state[i], zero_sm],
                 [wgt_meas[i].dot(wgt_state[i]), zero_mm]]
            ))
        # cholesky term
        chol_state = _semi_chol(var_state[i])
        chol_meas = _semi_chol(var_meas[i])
        chol_gm = chol_gm.at[i].set(jnp.block(
            [[chol_state, zero_sm],
             [wgt_meas[i].dot(chol_state), chol_meas]]
        ))
    return wgt_gm[1:], mu_gm, chol_gm

def _semi_chol(X):
    r"""
    Take cholesky for positive semi-definite matrices.
    Args:
        X (ndarray(n_dim, n_dim)): Variance matrix.
    
    Returns:
        (ndarray(n_dim, n_dim)): Cholesky decomposition of X.
        
    """
    ind = jnp.any(X, axis=1)
    jind = jnp.nonzero(ind)[0]
    nonzero_X = X[ind,]
    n_row = nonzero_X.shape[0]
    nonzero_X = nonzero_X[:n_row, :n_row]
    chol_X = jnp.zeros((len(ind), len(ind)))
    chol_X = chol_X.at[jnp.ix_(jind, jind)].set(jnp.linalg.cholesky(nonzero_X))
    return chol_X
