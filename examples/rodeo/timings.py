import warnings
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint
from numba import njit
from timeit import default_timer as timer
warnings.filterwarnings('ignore')
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, DirectAdjoint
from jax.config import config
config.update("jax_enable_x64", True)

from rodeo.ibm import ibm_init
from rodeo.ode import *

from ibm_nb import ibm_init as ibm_init_nb
from ibm_nb import indep_init
import ode_nb as rodeonb

# common helper function ---------------------------------------------------------------------------------------------
# definitions for logposterior
def _logpost(y_meas, Xt, gamma):
    return jnp.sum(jsp.stats.norm.logpdf(x=y_meas, loc=Xt, scale=gamma))

def logpost_rodeo(theta, gamma, fun):
    Xt = mv_jit(key, fun,
                 x0=x0_block, theta=theta,
                 tmin=tmin, tmax=tmax, n_steps=n_steps,
                 W=W_block, **ode_init, interrogate=interrogate_kramer)[0][:, :, 0]
    return _logpost(Xt, Xt, gamma)

def logpost_diffrax(theta, gamma, fun):
    term = ODETerm(fun)
    solver = Dopri5()
    saveat = SaveAt(ts=tseq)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    Xt = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                     stepsize_controller=stepsize_controller).ys
    return _logpost(Xt, Xt, gamma)

# timing helper functions
def rodeo_time(fun, n_loops, n_times):
    "Compute time required for rodeo"
    mv_jit(key, fun,
            x0=x0_block, theta=thetaj,
            tmin=tmin, tmax=tmax, n_steps=n_steps,
            W=W_block, **ode_init, interrogate=interrogate_kramer) # run once to jit-compile
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            mv_jit(key, fun,
                    x0=x0_block, theta=thetaj,
                    tmin=tmin, tmax=tmax, n_steps=n_steps,
                    W=W_block, **ode_init, interrogate=interrogate_kramer)
    end = timer()
    time_rd = (end - start)/n_loops/n_times

    # rodeo grad
    grad_jit1(thetaj, gamma, fun)
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            grad_jit1(thetaj, gamma, fun)
        end = timer()
    time_rdgrad = (end - start)/n_loops/n_times
    return time_rd, time_rdgrad

def rodeo_nb_time(fun, n_loops, n_times):
    "Compute time required for rodeo non-block"
    mv_jit2(key, fun,
            x0=x0_state, theta=thetaj,
            tmin=tmin, tmax=tmax, n_steps=n_steps,
            W=W, **ode_initnb, interrogate=rodeonb.interrogate_kramer) # run once to jit-compile
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            mv_jit2(key, fun,
                    x0=x0_state, theta=thetaj,
                    tmin=tmin, tmax=tmax, n_steps=n_steps,
                    W=W, **ode_initnb, interrogate=rodeonb.interrogate_kramer)
    end = timer()
    return (end - start)/n_loops/n_times

def odeint_time(fun, n_loops, n_times):
    "Compute time required for odeint"
    odeint(fun, ode0, tseq, args=(theta,)) # run once to jit-compile
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            odeint(fun, ode0, tseq, args=(theta,))
    end = timer()
    return (end - start)/n_loops/n_times

def diffrax_time(fun, n_loops, n_times):
    "Compute time required for diffrax"
    term = ODETerm(fun)
    solver = Dopri5()
    saveat = SaveAt(ts=tseq)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    diffeqsolve(term, solver, args = thetaj, t0=tmin, t1=tmax, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                stepsize_controller=stepsize_controller) # run once to jit-compile
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            diffeqsolve(term, solver, args = thetaj, t0=tmin, t1=tmax, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                        stepsize_controller=stepsize_controller)
    end = timer()
    time_rax = (end - start)/n_loops/n_times

    # diffrax grad
    grad_jit2(thetaj, gamma, fun)
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            grad_jit2(thetaj, gamma, fun)
    end = timer()
    time_raxgrad = (end - start)/n_loops/n_times
    return time_rax, time_raxgrad

# Jit solvers
key = jax.random.PRNGKey(0)
mv_jit = jax.jit(solve_mv, static_argnums=(1, 7, 10))
mv_jit2 = jax.jit(rodeonb.solve_mv, static_argnums=(1, 7, 10))
# jit grad for diffrax and rodeo
gamma = 0.1
grad_jit1 = jax.jit(jax.grad(logpost_rodeo), static_argnums=(2,))
grad_jit2 = jax.jit(jax.grad(logpost_diffrax), static_argnums=(2,))

# timing for each example starts here --------------------------------------------------------------------------------
# Chkrbetii ----------------------------------------------------------------------------------------------------------
def chkrebtii(X_t, t, theta):
    "Chkrebtii ODE."
    return jnp.array([[jnp.sin(2*t) - X_t[0, 0]]])

def chkrebtii_nb(X_t, t, theta):
    "Chkrebtii ODE."
    return jnp.array([jnp.sin(2*t) - X_t[0]])

@njit
def chkrebtii_ode(X_t, t, theta):
    "Jit for odeint to be on equal footing"
    return np.array([X_t[1], np.sin(2*t) - X_t[0]])

def chkrebtii_rax(t, X_t, theta):
    return jnp.array([X_t[1], jnp.sin(2*t) - X_t[0]])

# problem setup and intialization
n_obs = 1  # Total measures
n_deriv = jnp.array([4]*n_obs)

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_steps = 30
tmin = 0.
tmax = 10.
theta = thetaj = None
tseq = np.linspace(tmin, tmax, n_steps+1)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = .1
sigma = jnp.array([sigma]*n_obs)

# Initial W for rodeo
W_block = jnp.array([[[0.0, 0.0, 1.0, 0.0]]])

# Initial x0 for odeint
ode0 = np.array([-1., 0.])

# Initial x0 for rodeo
x0_block = jnp.array([[-1., 0., 1., 0.0]])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_steps
ode_init = ibm_init(dt, n_deriv, sigma)

# Initial W for non block
W = jnp.array([[0.0, 0.0, 1.0, 0.0]])

# Initial x0 for non block
x0_state = x0_block.flatten()

# Get parameters for non block
ode_init2 = ibm_init_nb(dt, n_deriv, sigma)
kinit = indep_init(ode_init2, n_deriv)
ode_initnb = dict((k, jnp.array(v)) for k, v in kinit.items())

# Timings
n_loops = 500
n_times = 10
time_rd, time_rdgrad = rodeo_time(chkrebtii, n_loops, n_times) # rodeo
time_rdnb = rodeo_nb_time(chkrebtii_nb, n_loops, n_times) # rodeo non-block
time_ode = odeint_time(chkrebtii_ode, n_loops, n_times) # odeint
time_rax, time_raxgrad = diffrax_time(chkrebtii_rax, n_loops, n_times) # diffrax

print("Chkrebtii----------------------------------")
print("rodeo / odeint = {}".format(time_ode/time_rd))
print("rodeo / diffrax = {}".format(time_rax/time_rd))
print("rodeo / non-blocking = {}".format(time_rdnb/time_rd))
print("rodeo grad / diffrax grad = {}".format(time_raxgrad/time_rdgrad))

# Fitz-Hugh Nagumo ---------------------------------------------------------------------------------------------------
# used by rodeo
def fitz(X_t, t, theta):
    "FitzHugh-Nagumo ODE."
    a, b, c = theta
    V, R = X_t[:,0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                    [-1/c*(V - a + b*R)]])

# used by rodeo non block
def fitz_nb(X, t, theta):
    "FitzHugh-Nagumo ODE function for jax."
    a, b, c = theta
    p = len(X)//2
    V, R = X[0], X[p]
    return jnp.array([c*(V - V*V*V/3 + R),
                      -1/c*(V - a + b*R)])

# used by odeint
@njit
def fitz_ode(X_t, t, theta):
    a, b, c = theta
    V, R = X_t
    return np.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

# used by diffrax
def fitz_rax(t, X_t, theta):
    a, b, c = theta
    V, R = X_t
    return jnp.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

# problem setup and intialization
n_deriv = 3  # Total state
n_obs = 2  # Total measures

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_steps = 250
tmin = 0.
tmax = 40.
theta = np.array([0.2, 0.2, 3])
thetaj = jnp.array(theta)
tseq = np.linspace(tmin, tmax, n_steps+1)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = .5
sigma = jnp.array([sigma]*n_obs)

# Initial W for rodeo
W_mat = np.zeros((n_obs, 1, n_deriv))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

# Initial x0 for odeint
ode0 = np.array([-1., 1.])

# Initial x0 for rodeo
x0_block = jnp.array([[-1., 1., 0.], [1., 1/3, 0.]])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_steps
n_order = jnp.array([n_deriv]*n_obs)
ode_init = ibm_init(dt, n_order, sigma)

# Initial W for rodeo non block
W = np.zeros((n_obs, jnp.sum(n_order)))
W[0, 1] = 1
W[1, n_deriv+1] = 1
W = jnp.array(W)

# Initial x0 for non block
x0_state = x0_block.flatten()

# parameters for non block
ode_init2 = ibm_init_nb(dt, n_order, sigma)
kinit = indep_init(ode_init2, n_order)
ode_initnb = dict((k, jnp.array(v)) for k, v in kinit.items())

# Timings
n_loops = 500
n_times = 10
time_rd, time_rdgrad = rodeo_time(fitz, n_loops, n_times) # rodeo
time_rdnb = rodeo_nb_time(fitz_nb, n_loops, n_times) # rodeo non-block
time_ode = odeint_time(fitz_ode, n_loops, n_times) # odeint
time_rax, time_raxgrad = diffrax_time(fitz_rax, n_loops, n_times) # diffrax

print("Fitz-Hugh----------------------------------")
print("rodeo / odeint = {}".format(time_ode/time_rd))
print("rodeo / diffrax = {}".format(time_rax/time_rd))
print("rodeo / non-blocking = {}".format(time_rdnb/time_rd))
print("rodeo grad / diffrax grad = {}".format(time_raxgrad/time_rdgrad))

# Hes1 ---------------------------------------------------------------------------------------------------------------
def hes1(X_t, t, theta):
    P, M, H = jnp.exp(X_t[:, 0])
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return jnp.array([[x1], [x2], [x3]])

def hes1_nb(X_t, t, theta):
    P, M, H = jnp.exp(X_t[::3])
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return jnp.array([x1, x2, x3])

@njit
def hes1_ode(X_t, t, theta):
    P, M, H = np.exp(X_t)
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return np.array([x1, x2, x3])

def hes1_rax(t, X_t, theta):
    P, M, H = jnp.exp(X_t)
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return jnp.array([x1, x2, x3])

# problem setup and intialization
n_deriv = 1  # Total state
n_obs = 3  # Total measures
n_deriv_prior = 3

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_steps = 120
tmin = 0.
tmax = 240.
theta = np.array([0.022, 0.3, 0.031, 0.028, 0.5, 20, 0.3])
thetaj = jnp.array(theta)
tseq = np.linspace(tmin, tmax, n_steps+1)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = 0.1
sigma = jnp.array([sigma]*n_obs)

# Initial value, x0, for the IVP
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

# Initial x0 for odeint
ode0 = np.log(np.array([1.439, 2.037, 17.904]))

# Initial x0 for jax block
x0 = jnp.log(jnp.array([[1.439], [2.037], [17.904]]))
v0 = hes1(x0, 0, theta)
X0 = jnp.concatenate([x0, v0],axis=1)
pad_dim = n_deriv_prior - n_deriv - 1
x0_block = jnp.pad(X0, [(0, 0), (0, pad_dim)])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_steps
n_order = jnp.array([n_deriv_prior]*n_obs)
ode_init = ibm_init(dt, n_order, sigma)

# Initial W for jax non block
W = np.zeros((n_obs, jnp.sum(n_order)))
for i in range(n_obs):
    W[i, n_deriv+i*n_deriv_prior] = 1
W = jnp.array(W)

# Initial x0 for non block
x0_state = x0_block.flatten()

# Get parameters for non block
ode_init2 = ibm_init_nb(dt, n_order, sigma)
kinit = indep_init(ode_init2, n_order)
ode_initnb = dict((k, jnp.array(v)) for k, v in kinit.items())

# Timings
n_loops = 500
n_times = 10
time_rd, time_rdgrad = rodeo_time(hes1, n_loops, n_times) # rodeo
time_rdnb = rodeo_nb_time(hes1_nb, n_loops, n_times) # rodeo non-block
time_ode = odeint_time(hes1_ode, n_loops, n_times) # odeint
time_rax, time_raxgrad = diffrax_time(hes1_rax, n_loops, n_times) # diffrax

print("Hes1----------------------------------")
print("rodeo / odeint = {}".format(time_ode/time_rd))
print("rodeo / diffrax = {}".format(time_rax/time_rd))
print("rodeo / non-blocking = {}".format(time_rdnb/time_rd))
print("rodeo grad / diffrax grad = {}".format(time_raxgrad/time_rdgrad))

# SEIRAH -------------------------------------------------------------------------------------------------------------
def seirah(X_t, t, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X_t[:, 0]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q= theta
    D_h = 30
    x1 = -b*S*(I + alpha*A)/N
    x2 = b*S*(I + alpha*A)/N - E/D_e
    x3 = r*E/D_e - I/D_q - I/D_I
    x4 = (I + A)/D_I + H/D_h
    x5 = (1-r)*E/D_e - A/D_I
    x6 = I/D_q - H/D_h
    return jnp.array([[x1], [x2], [x3], [x4], [x5], [x6]])

def seirah_nb(X_t, t, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X_t[::3]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q= theta
    D_h = 30
    x1 = -b*S*(I + alpha*A)/N
    x2 = b*S*(I + alpha*A)/N - E/D_e
    x3 = r*E/D_e - I/D_q - I/D_I
    x4 = (I + A)/D_I + H/D_h
    x5 = (1-r)*E/D_e - A/D_I
    x6 = I/D_q - H/D_h
    return jnp.array([x1, x2, x3, x4, x5, x6])

@njit
def seirah_ode(X_t, t, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X_t
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b*S*(I + alpha*A)/N
    dE = b*S*(I + alpha*A)/N - E/D_e
    dI = r*E/D_e - I/D_q - I/D_I
    dR = (I + A)/D_I + H/D_h
    dA = (1-r)*E/D_e - A/D_I
    dH = I/D_q - H/D_h
    out = np.array([dS, dE, dI, dR, dA, dH])
    return out

def seirah_rax(t, X_t, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X_t
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b*S*(I + alpha*A)/N
    dE = b*S*(I + alpha*A)/N - E/D_e
    dI = r*E/D_e - I/D_q - I/D_I
    dR = (I + A)/D_I + H/D_h
    dA = (1-r)*E/D_e - A/D_I
    dH = I/D_q - H/D_h
    out = jnp.array([dS, dE, dI, dR, dA, dH])
    return out

# problem setup and intialization
n_deriv = 1  # Total state
n_obs = 6  # Total measures
n_deriv_prior = 3

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_steps = 80
tmin = 0.
tmax = 60.
theta = np.array([2.23, 0.034, 0.55, 5.1, 2.3, 0.36])
thetaj = jnp.array(theta)
tseq = np.linspace(tmin, tmax, n_steps+1)

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = jnp.array([.5]*n_obs)

# W matrix for the IVP
W_mat = np.zeros((n_obs, 1, n_deriv_prior))
W_mat[:, :, 1] = 1
W_block = jnp.array(W_mat)

# Initial x0 for odeint
ode0 = np.array([63804435., 15492., 21752., 0., 618013., 93583.])

# Initial x0 for jax block
x0 = jnp.array([[63804435], [15492], [21752], [0], [618013], [93583]])
v0 = seirah(x0, 0, theta)
X0 = jnp.concatenate([x0, v0],axis=1)
pad_dim = n_deriv_prior - n_deriv - 1
x0_block = jnp.pad(X0, [(0, 0), (0, pad_dim)])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_steps
n_order = jnp.array([n_deriv_prior]*n_obs)
ode_init = ibm_init(dt, n_order, sigma)

# Initial W for jax non block
W = np.zeros((n_obs, jnp.sum(n_order)))
for i in range(n_obs):
    W[i, n_deriv+i*n_deriv_prior] = 1
W = jnp.array(W)

# Initial x0 for non block
x0_state = x0_block.flatten()

# Get parameters for non block
ode_init2 = ibm_init_nb(dt, n_order, sigma)
kinit = indep_init(ode_init2, n_order)
ode_initnb = dict((k, jnp.array(v)) for k, v in kinit.items())

# Timings
n_loops = 500
n_times = 10
time_rd, time_rdgrad = rodeo_time(seirah, n_loops, n_times) # rodeo
time_rdnb = rodeo_nb_time(seirah_nb, n_loops, n_times) # rodeo non-block
time_ode = odeint_time(seirah_ode, n_loops, n_times) # odeint
time_rax, time_raxgrad = diffrax_time(seirah_rax, n_loops, n_times) # diffrax

print("SEIRAH----------------------------------")
print("rodeo / odeint = {}".format(time_ode/time_rd))
print("rodeo / diffrax = {}".format(time_rax/time_rd))
print("rodeo / non-blocking = {}".format(time_rdnb/time_rd))
print("rodeo grad / diffrax grad = {}".format(time_raxgrad/time_rdgrad))
