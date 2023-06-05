from timeit import default_timer as timer
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import odeint
from numba import njit
warnings.filterwarnings('ignore')
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

from rodeo.ibm import ibm_init
from rodeo.ode import *

from ibm_nb import ibm_init as ibm_init_nb
from ibm_nb import indep_init
import ode_nb as rodeonb

def ode_fun_jax(X_t, t, theta):
    "Chkrebtii ODE."
    return jnp.array([[jnp.sin(2*t) - X_t[0, 0]]])

# ode function used by jax non block
def ode_fun_jax2(X_t, t, theta):
    "Chkrebtii ODE."
    return jnp.array([jnp.sin(2*t) - X_t[0]])

@njit
def ode_fun(X_t, t, theta):
    return np.array([X_t[1], np.sin(2*t) - X_t[0]])

def ode_fun_rax(t, X_t, theta):
    return jnp.array([X_t[1], jnp.sin(2*t) - X_t[0]])

def _logpost(y_meas, Xt, gamma):
    return jnp.sum(jsp.stats.norm.logpdf(x=y_meas, loc=Xt, scale=gamma))

def logpost_rodeo(theta, y_meas, gamma):
    Xt = solve_sim(key=key, fun=ode_fun_jax,
                    x0=x0_block, theta=theta,
                    tmin=tmin, tmax=tmax, n_steps=n_steps,
                    W=W_block, **ode_init)
    return _logpost(y_meas, Xt[:,:,0], gamma)

def logpost_diffrax(theta, y_meas, gamma):
    Xt = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                      stepsize_controller=stepsize_controller).ys
    return _logpost(y_meas, Xt, gamma)

# problem setup and intialization
n_obs = 1  # Total measures
n_deriv = jnp.array([4]*n_obs)

# it is assumed that the solution is sought on the interval [tmin, tmax].
n_steps = 80
tmin = 0.
tmax = 10.
theta = thetaj = None

# The rest of the parameters can be tuned according to ODE
# For this problem, we will use
sigma = .1
sigma = jnp.array([sigma]*n_obs)

# Initial W for jax block
W_block = jnp.array([[[0.0, 0.0, 1.0, 0.0]]])

# Initial x0 for odeint
ode0 = np.array([-1., 0.])

# Initial x0 for jax block
x0_block = jnp.array([[-1., 0., 1., 0.0]])

# Get parameters needed to run the solver
dt = (tmax-tmin)/n_steps
ode_init = ibm_init(dt, n_deriv, sigma)

# Initial W for jax non block
W = jnp.array([[0.0, 0.0, 1.0, 0.0]])

# Initial x0 for non block
x0_state = x0_block.flatten()

# Ger parameters for non block
ode_init2 = ibm_init_nb(dt, n_deriv, sigma)
kinit = indep_init(ode_init2, n_deriv)
ode_initnb = dict((k, jnp.array(v)) for k, v in kinit.items())

# Jit solver
key = jax.random.PRNGKey(0)
sim_jit = jax.jit(solve_sim, static_argnums=(1, 7))
sim_jit(key=key, fun=ode_fun_jax,
        x0=x0_block, theta=thetaj,
        tmin=tmin, tmax=tmax, n_steps=n_steps,
        W=W_block, **ode_init)

# Jit non block solver
sim_jit2 = jax.jit(rodeonb.solve_sim, static_argnums=(1, 7))
sim_jit2(key=key, fun=ode_fun_jax2,
         x0=x0_state, theta=thetaj,
         tmin=tmin, tmax=tmax, n_steps=n_steps,
         W=W, **ode_initnb) 

# Timings
n_loops = 100

# Jax block
start = timer()
for i in range(n_loops):
    _ = sim_jit(key=key, fun=ode_fun_jax,
                x0=x0_block, theta=thetaj,
                tmin=tmin, tmax=tmax, n_steps=n_steps,
                W=W_block, **ode_init)
end = timer()
time_jax = (end - start)/n_loops

# Jax non block
start = timer()
for i in range(n_loops):
    _ = sim_jit2(key=key, fun=ode_fun_jax2,
                 x0=x0_state, theta=thetaj,
                 tmin=tmin, tmax=tmax, n_steps=n_steps,
                 W=W, **ode_initnb)
end = timer()
time_jaxnb = (end - start)/n_loops

# odeint
tseq = np.linspace(tmin, tmax, n_steps+1)
y_meas = odeint(ode_fun, ode0, tseq, args=(theta,))
start = timer()
for i in range(n_loops):
    _ = odeint(ode_fun, ode0, tseq, args=(theta,))
end = timer()
time_ode = (end - start)/n_loops

# # diffrax
tseq = np.linspace(tmin, tmax, n_steps+1)
term = ODETerm(ode_fun_rax)
solver = Dopri5()
saveat = SaveAt(ts=tseq)
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
sol = diffeqsolve(term, solver, args = thetaj, t0=tmin, t1=tmax, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                  stepsize_controller=stepsize_controller)
start = timer()
for i in range(n_loops):
    _ = diffeqsolve(term, solver, args = thetaj, t0=tmin, t1=tmax, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                    stepsize_controller=stepsize_controller)
end = timer()
time_rax = (end - start)/n_loops

# jit grad for diffrax and rodeo
gamma = 0.1
grad_jit1 = jax.jit(jax.grad(logpost_rodeo))
grad_jit2 = jax.jit(jax.grad(logpost_diffrax))

# rodeo grad
start = timer()
for i in range(n_loops):
    _ = grad_jit1(thetaj, y_meas, gamma)
end = timer()
time_jaxgrad = (end - start)/n_loops

# diffrax grad
start = timer()
for i in range(n_loops):
    _ = grad_jit2(thetaj, y_meas, gamma)
end = timer()
time_raxgrad = (end - start)/n_loops

print("Number of times faster jax is compared to odeint {}".format(time_ode/time_jax))
print("Number of times faster jax is compared to diffrax {}".format(time_rax/time_jax))
print("Number of times faster jax is compared to non-blocking {}".format(time_jaxnb/time_jax))
print("Number of times faster jax is compared to diffrax for grad {}".format(time_raxgrad/time_jaxgrad))
