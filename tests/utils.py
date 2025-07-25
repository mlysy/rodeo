import unittest
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from rodeo.prior import ibm_init
from rodeo.utils import mvncond
import gauss_markov as gm

def rel_err(X1, X2):
    """
    Relative error between two JAX arrays.
    Adds 0.1 to the denominator to avoid nan's when its equal to zero.
    """
    x1 = X1.ravel() * 1.0
    x2 = X2.ravel() * 1.0
    return jnp.max(jnp.abs((x1 - x2)/(0.1 + x1)))

def abs_err(X1, X2):
    return jnp.max(jnp.abs(X1 - X2))


def kalman_theta(m, y, mu, Sigma):
    """
    Calculate theta{m|n} using the joint density.

    Args:
        m (ndarray(n_x)): State variable indices.
        y (ndarray(n+1,n_meas)): Measurement variable observations.
        mu (ndarray(k,n_dim)): Joint mean, where `k >= max(m, n) + 1`.
        Sigma (ndarray(k,n_dim,k,n_dim)): Joint variance.

    Returns:
        (tuple):
        **mean_cond** (ndarray(n_x, n_state)): E[x_m | y_0:n].
        **var_cond** (ndarray(n_x, n_state, n_x, n_state)): var(x_m | y_0:n).
        **Note:** In both cases if `n_x == 1` the corresponding dimension is squeezed.
    """
    # dimensions
    n_tot, n_dim = mu.shape
    n_y, n_meas = y.shape
    n_state = n_dim - n_meas
    m = np.atleast_1d(m)
    n_x = len(m)
    # conditioning indices
    icond = np.full((n_tot, n_dim), False)
    icond[:n_y, n_state:n_dim] = True
    # marginal indices on the flattened scale
    imarg = np.full((n_tot, n_dim), False)
    imarg[np.ix_(m, np.arange(n_state))] = True
    imarg = np.ravel(imarg)[~np.ravel(icond)]
    A, b, V = mvncond(mu=np.ravel(mu),
                      Sigma=np.reshape(Sigma, (n_tot*n_dim, n_tot*n_dim)),
                      icond=np.ravel(icond))
    mu_mn = (A.dot(np.ravel(y)) + b)[imarg]
    V_mn = V[np.ix_(imarg, imarg)]
    mu_mn = mu_mn.reshape((n_x, n_state))
    V_mn = V_mn.reshape((n_x, n_state, n_x, n_state))
    if n_x == 1:
        mu_mn = mu_mn.squeeze(axis=(0,))
        V_mn = V_mn.squeeze(axis=(0, 2))
    return mu_mn, V_mn

def fitz_setup(self):
    n_vars = 2  # Number of variables
    n_deriv = 3

    # it is assumed that the solution is sought on the interval [t_min, t_max].
    self.t_min = 0.
    self.t_max = 10.
    h = .05
    self.n_steps = int((self.t_max-self.t_min)/h)
    self.t = jnp.array(.25)  # time
    self.tseq = np.linspace(self.t_min, self.t_max, self.n_steps+1)

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    sigma = .001

    # Initial value, x0, for the IVP
    self.x0 = np.array([-1., 1.])
    self.x0_block = jnp.array([[-1., 1, 0], [1., 1/3, 0]])

    # function parameter
    self.theta = jnp.array([0.2, 0.2, 3])  # True theta
    sigma = jnp.array([sigma]*n_vars)
    self.prior_pars = ibm_init(h, n_deriv, sigma)
    self.prior_Q, self.prior_R = self.prior_pars
    
    # block
    n_bmeas = 1
    n_bstate = 3

    W_block = np.zeros((n_vars, n_bmeas, n_bstate))
    W_block[:, :, 1] = 1
    self.W_block = jnp.array(W_block)
    self.key = jax.random.PRNGKey(0)

    def fitz_jax(X_t, t, **params):
        "FitzHugh-Nagumo ODE."
        theta = params['theta']
        a, b, c = theta
        V, R = X_t[0, 0], X_t[1,0]
        return jnp.array([[c*(V - V*V*V/3 + R)],
                        [-1/c*(V - a + b*R)]])
    self.fitz_jax = fitz_jax

    def fitz_odeint(X_t, t, theta):
        a, b, c = theta
        V, R = X_t
        return np.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

    self.fitz_odeint = fitz_odeint


def kalman_setup(self):
    key = random.PRNGKey(0)
    key, *subkey = random.split(key, 3)
    self.n_meas = random.randint(subkey[0], (1,), 1, 4)[0]
    self.n_state = int(self.n_meas + random.randint(subkey[1], (1,), 1, 5)[0])
    self.n_tot = 3
    # self.n_meas = 3
    # self.n_state = 4

    self.key, *subkeys = random.split(key, 10)
    self.mean_state = random.normal(subkeys[0], (self.n_tot, self.n_state))
    self.var_state = random.normal(
        subkeys[1], (self.n_tot, self.n_state, self.n_state))
    self.var_state = jax.vmap(lambda vs: vs.dot(vs.T))(self.var_state)
    self.wgt_state = 0.01 * \
        random.normal(subkeys[2], (self.n_tot-1, self.n_state, self.n_state))
    # wgt_state = jnp.zeros((n_tot-1, self.n_state, self.n_state))
    self.mean_meas = random.normal(subkeys[3], (self.n_tot, self.n_meas,))
    self.var_meas = random.normal(
        subkeys[4], (self.n_tot, self.n_meas, self.n_meas))
    self.var_meas = jax.vmap(lambda vs: vs.dot(vs.T))(self.var_meas)
    self.wgt_meas = random.normal(
        subkeys[5], (self.n_tot, self.n_meas, self.n_state))
    # wgt_meas = jnp.zeros((n_tot, n_meas, self.n_state))
    self.x_meas = random.normal(subkeys[6], (self.n_tot, self.n_meas))
    self.x_state_next = random.normal(subkeys[7], (self.n_state,))
    self.z_state = random.normal(subkeys[8], (self.n_state,))

    A_gm, b_gm, C_gm = gm.kalman2gm(
        wgt_state=self.wgt_state,
        mean_state=self.mean_state,
        var_state=self.var_state,
        wgt_meas=self.wgt_meas,
        mean_meas=self.mean_meas,
        var_meas=self.var_meas
    )

    self.mean_gm, self.var_gm = gm.gauss_markov_mv(A=A_gm, b=b_gm, C=C_gm)


def test_filter(self):
    # theta_{0|0}
    mean_state_past, var_state_past = kalman_theta(
        m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
    )
    # theta_{1|0}
    mean_state_pred1, var_state_pred1 = kalman_theta(
        m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
    )
    # theta_{1|1}
    mean_state_filt1, var_state_filt1 = kalman_theta(
        m=1, y=self.x_meas[0:2], mu=self.mean_gm, Sigma=self.var_gm
    )
    return mean_state_past, var_state_past, mean_state_pred1, var_state_pred1, mean_state_filt1, var_state_filt1


def test_filter2(self):
    # theta_{1|1}
    mean_state_past, var_state_past = kalman_theta(
        m=1, y=self.x_meas[0:2], mu=self.mean_gm, Sigma=self.var_gm
    )
    # theta_{2|1}
    mean_state_pred1, var_state_pred1 = kalman_theta(
        m=2, y=self.x_meas[0:2], mu=self.mean_gm, Sigma=self.var_gm
    )
    # theta_{2|2}
    mean_state_filt1, var_state_filt1 = kalman_theta(
        m=2, y=self.x_meas[0:3], mu=self.mean_gm, Sigma=self.var_gm
    )

    return mean_state_past, var_state_past, mean_state_pred1, var_state_pred1, mean_state_filt1, var_state_filt1 


def test_smooth(self):
    # theta_{1|1}
    mean_state_next, var_state_next = kalman_theta(
        m=1, y=self.x_meas, mu=self.mean_gm, Sigma=self.var_gm
    )
    # theta_{0|0}
    mean_state_filt, var_state_filt = kalman_theta(
        m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
    )
    # theta_{1|0}
    mean_state_pred, var_state_pred = kalman_theta(
        m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
    )
    # theta_{0:1|1}
    mean_state_smooth1, var_state_smooth1 = kalman_theta(
        m=[0, 1], y=self.x_meas, mu=self.mean_gm, Sigma=self.var_gm
    )
    A, b, V = mvncond(
        mu=mean_state_smooth1.ravel(),
        Sigma=var_state_smooth1.reshape(2*self.n_state, 2*self.n_state),
        icond=jnp.array([False]*self.n_state + [True]*self.n_state)
    )
    mean_state_sim1 = A.dot(self.x_state_next)+b
    
    return mean_state_next, var_state_next, mean_state_filt, var_state_filt, mean_state_pred, \
                var_state_pred, mean_state_smooth1, var_state_smooth1, mean_state_sim1, V, A, b

