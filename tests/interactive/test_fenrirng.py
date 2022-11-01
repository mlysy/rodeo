import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint

from rodeo.ibm_init import ibm_init
from rodeo.ode_solve import *
from fenrir_filter import *
from ibm_nb_init import ibm_init as ibm_init_nb
from ibm_nb_init import indep_init
from jax.config import config
config.update("jax_enable_x64", True)

def fitz(X_t, t, theta):
    "Fitz ODE written for jax"
    a, b, c = theta
    V, R = X_t[:, 0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                      [-1/c*(V - a + b*R)]])

def fitz_ode(X_t, t, theta):
    "Fitz ODE written for odeint"
    a, b, c = theta
    p = len(X_t)//2
    V, R = X_t[0], X_t[p]
    return jnp.array([c*(V - V*V*V/3 + R),
                    -1/c*(V - a + b*R)])


def obs_fun(y, Cx, gamma, theta):
    mu_obs, var_obs = gamma
    mu = Cx + mu_obs
    return jsp.stats.multivariate_normal.logpdf(y, mu, var_obs)

def x_fun(x, y, theta):
    return x

"Perform parameter inference using the FitzHugh-Nagumo function."
# These parameters define the order of the ODE and the CAR(p) process
n_deriv = 1 # number of derivatives in IVP
n_obs = 2 # number of observations.
n_deriv_prior = 3 # number of derivatives in IBM prior

# it is assumed that the solution is sought on the interval [tmin, tmax].
tmin = 0.
tmax = 40.
n_eval = 400
n_res = int(n_eval/tmax)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = jnp.array([.1]*n_obs)
n_order = jnp.array([n_deriv_prior]*n_obs)

# Initial x0 for odeint
ode0 = np.array([-1., 1.])

# Initial value, x0, for the IVP
x0_block = jnp.array([[-1., 1., 0.], [1., 1/3, 0.]])
x0_state = x0_block.flatten()

# Initial W for jax block
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

# Initial W for jax non block
W = np.zeros((n_obs, jnp.sum(n_order)))
W[0, 1] = 1
W[1, n_deriv_prior+1] = 1
W = jnp.array(W)

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_eval
n_order = jnp.array([n_deriv_prior]*n_obs)
ode_init = ibm_init(dt, n_order, sigma)

# Ger parameters for non block
ode_init2 = ibm_init_nb(dt, n_order, sigma)
kinit = indep_init(ode_init2, n_order)
ode_initnb = dict((k, jnp.array(v)) for k, v in kinit.items())

# logprior parameters
theta = np.array([0.2, 0.2, 3]) # True theta

# Observation noise
gamma = 0.2

# parameters for fenrir_filter
mu_obs = jnp.zeros((n_obs,))
wgt_obs = np.zeros((n_obs, jnp.sum(n_order)))
wgt_obs[0,0] = 1
wgt_obs[1, n_deriv_prior] = 1
wgt_obs = jnp.array(wgt_obs)
var_obs = gamma**2*jnp.eye((n_obs))

# Initialize inference class and simulate observed data
key = jax.random.PRNGKey(0)
tseq = np.linspace(tmin, tmax, 41)
X_t = odeint(fitz_ode, ode0, tseq, args=(theta,))
e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=X_t.shape)
Y_t = X_t + gamma*e_t

# jit solver
key = jax.random.PRNGKey(0)
rodeo_jit = jax.jit(solve_mv, static_argnums=(1, 6))
df_jit = jax.jit(fenrir_filter, static_argnums=(0, 5))
ng_jit = jax.jit(fenrir_filterng, static_argnums=(0, 5, 13, 14))

# Jit solver
m_rodeo = rodeo_jit(key=key, fun=fitz,
        x0=x0_block, theta=theta,
        tmin=tmin, tmax=tmax, n_eval=n_eval,
        wgt_meas=W_block, **ode_init)[0][::n_res, :, 0]

l_rodeo = jnp.sum(jsp.stats.multivariate_normal.logpdf(Y_t, m_rodeo, gamma**2))
l_fen = df_jit(fitz_ode, x0_state, theta, tmin, tmax, n_res, W,
                ode_initnb["wgt_state"], ode_initnb["mu_state"], ode_initnb["var_state"],
                wgt_obs, mu_obs, var_obs, Y_t)
l_fenng = ng_jit(fitz_ode, x0_state, theta, tmin, tmax, n_res, W,
                 ode_initnb["wgt_state"], ode_initnb["mu_state"], ode_initnb["var_state"],
                 wgt_obs, Y_t, (mu_obs, var_obs), obs_fun, x_fun)

print("Difference between rodeo and fenrir {}".format(l_rodeo - l_fen))
print("Difference between fenrir and fenrirng {}".format(l_fen - l_fenng))
