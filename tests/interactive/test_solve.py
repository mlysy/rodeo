from functools import partial
from timeit import timeit

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=2)
def solve_var(a, b, method):
    """
    Solve for a symmetric positive semidefine matrix a.
    """
    if method == "gen":
        return jax.scipy.linalg.solve(a, b, assume_a="gen")
    elif method == "sym":
        return jax.scipy.linalg.solve(a, b, assume_a="sym")
    elif method == "her":
        return jax.scipy.linalg.solve(a, b, assume_a="her")
    elif method == "pos":
        return jax.scipy.linalg.solve(a, b, assume_a="pos")
    elif method == "chol":
        cfac = jax.scipy.linalg.cho_factor(a)
        return jax.scipy.linalg.cho_solve(cfac, b)
    elif method == "pinv":
        return jnp.dot(jnp.linalg.pinv(a, hermitian=True), b)
    elif method == "eigh":
        vals, vecs = jnp.linalg.eigh(a)
        z = jnp.dot(vecs.T, b)
        not_zero = ~jnp.isclose(vals, 0.0, rtol=1e-10)
        new_vals = jnp.where(not_zero, vals, 1.0)
        inv_vals = jnp.where(not_zero, 1.0 / new_vals, 0.0)
        return jnp.linalg.multi_dot([vecs, jnp.diag(inv_vals), z])
    else:
        raise NotImplementedError


n_var = 6
n_zero = 3
n_eq = 1

key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, num=3)
a = jax.random.normal(subkeys[0], shape=(n_var - n_zero, n_var - n_zero))
a = a @ a.T
if n_zero > 0:
    key, subkey = jax.random.split(key)
    nonzero_ind = jax.random.choice(
        subkey, a=jnp.arange(n_var), shape=(n_var - n_zero,), replace=False
    )
    nonzero_ind = jnp.sort(nonzero_ind)
    zero_bool = jnp.full(shape=(n_var,), fill_value=True).at[nonzero_ind].set(False)
    A = jnp.zeros(shape=(n_var, n_var))
    A = A.at[jnp.ix_(nonzero_ind, nonzero_ind)].set(a)
    A = A.at[zero_bool, zero_bool].set(jnp.inf)
else:
    A = a
# if n_zero > 0:
#     a = jax.scipy.linalg.block_diag([a, jnp.zeros((n_zero, n_zero))])
b = jax.random.normal(subkeys[1], shape=(n_var, n_eq))

methods = ["gen", "sym", "her", "pos", "chol", "pinv", "eigh"]

[solve_var(a, b, method=m) for m in methods]

results = [
    timeit(f"solve_var(a, b, method='{m}')", globals=globals(), number=100_000)
    for m in methods
]
results = jnp.array(results)


# --- cholesky with infs -------------------------------------------------------


# --- scratch ------------------------------------------------------------------

dict(zip(methods, results))


def scan_fun(carry, x):
    y = jnp.sum(jnp.diag(jnp.arange(x)))
    carry = carry + y
    return carry, y


jax.lax.scan(scan_fun, init=jnp.array(0.0), xs=jnp.arange(5))
