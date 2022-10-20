import rodeo
import rodeo.gauss_markov as gm
from fenrir_filter import *
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.config import config
config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(0)

n_meas = 3
n_obs = 2
n_state = 4
n_tot = 3

key, *subkeys = jax.random.split(key, 14)
# states
b = jax.random.normal(subkeys[0], (n_tot, n_state))
V = jax.random.normal(subkeys[1], (n_tot, n_state, n_state))
V = jax.vmap(lambda vs: vs.dot(vs.T))(V)
C = jax.vmap(lambda vs: jnp.linalg.cholesky(vs))(V)
A = jax.random.normal(subkeys[2], (n_tot-1, n_state, n_state))
# obs
_mu_obs = jax.random.normal(subkeys[3], (n_obs,))
mu_obs = jnp.repeat(_mu_obs[jnp.newaxis], n_tot, 0)
_var_obs = jax.random.normal(subkeys[4], (n_obs, n_obs))
_var_obs = _var_obs.dot(_var_obs.T)
var_obs = jnp.repeat(_var_obs[jnp.newaxis], n_tot, 0)
_wgt_obs = jax.random.normal(subkeys[5], (n_obs, n_state))
wgt_obs = jnp.repeat(_wgt_obs[jnp.newaxis], n_tot, 0)
y_obs = jax.random.normal(subkeys[6], (n_tot, n_obs))

# fenrir
logdens = reverse(
    wgt_state=A, mu_state=b, 
    var_state=V, wgt_obs=_wgt_obs,
    mu_obs=_mu_obs, var_obs=_var_obs, y_obs=y_obs
)

# gauss markov
A_gm, b_gm, C_gm = gm.kalman2gm(
    wgt_state=A[::-1],
    mu_state=b[::-1],
    var_state=V[::-1],
    wgt_meas=wgt_obs[::-1],
    mu_meas=mu_obs[::-1],
    var_meas=var_obs[::-1]
)

mu_gm, V_gm = gm.gauss_markov_mv(A=A_gm, b=b_gm, C=C_gm)
mu_y = mu_gm[:, n_state:]
var_y = V_gm[:, n_state:, :, n_state:]
Sigma_y = np.reshape(var_y, (n_tot*n_obs, n_tot*n_obs))
logdens2 = jsp.stats.multivariate_normal.logpdf(y_obs[::-1].flatten(), mu_y.flatten(), Sigma_y)
print("Difference between logdensity: {}".format(logdens - logdens2))

# n_res > 1

n_res = 3
key, *subkeys = jax.random.split(key, 14)
# states
b = jax.random.normal(subkeys[0], ((n_tot-1)*n_res+1, n_state))
V = jax.random.normal(subkeys[1], ((n_tot-1)*n_res+1, n_state, n_state))
V = jax.vmap(lambda vs: vs.dot(vs.T))(V)
C = jax.vmap(lambda vs: jnp.linalg.cholesky(vs))(V)
A = jax.random.normal(subkeys[2], ((n_tot-1)*n_res, n_state, n_state))
# obs
_mu_obs = jax.random.normal(subkeys[3], (n_obs,))
mu_obs = jnp.repeat(_mu_obs[jnp.newaxis], (n_tot-1)*n_res+1, 0)
_var_obs = jax.random.normal(subkeys[4], (n_obs, n_obs))
_var_obs = _var_obs.dot(_var_obs.T)
var_obs = jnp.repeat(_var_obs[jnp.newaxis], (n_tot-1)*n_res+1, 0)
_wgt_obs = jax.random.normal(subkeys[5], (n_obs, n_state))
wgt_obs = jnp.repeat(_wgt_obs[jnp.newaxis], (n_tot-1)*n_res+1, 0)
y_obs = jax.random.normal(subkeys[6], (n_tot, n_obs))
y_out = jnp.ones(((n_tot-1)*n_res+1, n_obs))*jnp.nan
for i in range(n_tot):
    y_out = y_out.at[i*n_res].set(y_obs[i])

# fenrir
logdens = reverse(
    wgt_state=A, mu_state=b, 
    var_state=V, wgt_obs=_wgt_obs,
    mu_obs=_mu_obs, var_obs=_var_obs, y_obs=y_out
)

# gauus markov
A_gm, b_gm, C_gm = gm.kalman2gm(
    wgt_state=A[::-1],
    mu_state=b[::-1],
    var_state=V[::-1],
    wgt_meas=wgt_obs[::-1],
    mu_meas=mu_obs[::-1],
    var_meas=var_obs[::-1]
)

mu_gm, V_gm = gm.gauss_markov_mv(A=A_gm, b=b_gm, C=C_gm)
mu_y = mu_gm[::n_res, n_state:]
var_y = V_gm[::n_res, n_state:, ::n_res, n_state:]
Sigma_y = np.reshape(var_y, (n_tot*n_obs, n_tot*n_obs))
logdens2 = jsp.stats.multivariate_normal.logpdf(y_obs[::-1].flatten(), mu_y.flatten(), Sigma_y)

print("Difference between logdensity for n_res > 1: {}".format(logdens - logdens2))
