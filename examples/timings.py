import warnings
warnings.filterwarnings('ignore')
import numpy as np
import jax
import jax.numpy as jnp
# import jax.scipy as jsp
import jax.scipy.linalg as jsl
from scipy.integrate import odeint
from numba import njit
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
from timeit import default_timer as timer
from jax import config
config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

from rodeo.prior import ibm_init, indep_init
from rodeo.solve import solve_mv
from rodeo.interrogate import interrogate_kramer
import solve_nb

# helper function to help with the padding in rodeo format
from rodeo.utils import first_order_pad


# common helper function ---------------------------------------------------------------------------------------------


def rodeo_time(ode_fun, n_loops, n_times):
    "Compute time required for rodeo"
    mv_jit(key=key, ode_fun=ode_fun,
           ode_weight=W_block, ode_init=x0_block,
           t_min=t_min, t_max=t_max, n_steps=n_steps,
           interrogate=interrogate_kramer,
           prior_weight=prior_Q_block, prior_var=prior_R_block,
           theta=theta) # run once to jit-compile
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            mv_jit(key=key, ode_fun=ode_fun,
                   ode_weight=W_block, ode_init=x0_block,
                   t_min=t_min, t_max=t_max, n_steps=n_steps,
                   interrogate=interrogate_kramer,
                   prior_weight=prior_Q_block, prior_var=prior_R_block,
                   theta=theta) # run once to jit-compile
    end = timer()
    return (end - start)/n_loops/n_times


def rodeo_nb_time(ode_fun, n_loops, n_times):
    "Compute time required for rodeo non-block"
    mvnb_jit(key=key, ode_fun=ode_fun,
             ode_weight=W, ode_init=x0,
             t_min=t_min, t_max=t_max, n_steps=n_steps,
             interrogate=solve_nb.interrogate_kramer,
             prior_weight=prior_Q, prior_var=prior_R,
             theta=theta) # run once to jit-compile) # run once to jit-compile
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            mvnb_jit(key=key, ode_fun=ode_fun,
                     ode_weight=W, ode_init=x0,
                     t_min=t_min, t_max=t_max, n_steps=n_steps,
                     interrogate=solve_nb.interrogate_kramer,
                     prior_weight=prior_Q, prior_var=prior_R,
                     theta=theta) # run once to jit-compile) # run once to jit-compile
    end = timer()
    return (end - start)/n_loops/n_times


def odeint_time(ode_fun, n_loops, n_times):
    "Compute time required for odeint"
    odeint(ode_fun, ode0, tseq, args=(thetanp,)) # run once to jit-compile
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            odeint(ode_fun, ode0, tseq, args=(thetanp,))
    end = timer()
    return (end - start)/n_loops/n_times


def diffrax_time(ode_fun, n_loops, n_times):
    "Compute time required for diffrax"
    term = ODETerm(ode_fun)
    solver = Dopri5()
    saveat = SaveAt(ts=tseq)
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    diffeqsolve(term, solver, args=theta, t0=t_min, t1=t_max, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                stepsize_controller=stepsize_controller) # run once to jit-compile
    start = timer()
    for t in range(n_times):
        for i in range(n_loops):
            diffeqsolve(term, solver, args=theta, t0=t_min, t1=t_max, dt0=dt, y0=jnp.array(ode0), saveat=saveat,
                        stepsize_controller=stepsize_controller)
    end = timer()
    return (end - start)/n_loops/n_times

# Jit solvers
key = jax.random.PRNGKey(0)
mv_jit = jax.jit(solve_mv, static_argnums=(1, 6, 7))
mvnb_jit = jax.jit(solve_nb.solve_mv, static_argnums=(1, 6, 7))

# timing for each example starts here --------------------------------------------------------------------------------
# Chkrbetii ----------------------------------------------------------------------------------------------------------
def chkrebtii(X, t, theta):
    "Chkrebtii ODE."
    return jnp.array([[jnp.sin(2*t) - X[0, 0]]])

def chkrebtii_nb(X, t, theta):
    "Chkrebtii ODE non-block."
    return jnp.array([jnp.sin(2*t) - X[0]])

@njit
def chkrebtii_ode(X, t, theta):
    "Jit for odeint to be on equal footing"
    return np.array([X[1], np.sin(2*t) - X[0]])

def chkrebtii_rax(t, X, theta):
    "Chkrebtii ODE for diffrax"
    return jnp.array([X[1], jnp.sin(2*t) - X[0]])

W_block = jnp.array([[[0., 0., 1., 0.]]])  # LHS matrix of ODE
W = W_block[0]
x0_block = jnp.array([[-1., 0., 1., 0.]])  # initial value for the IVP
x0 = x0_block[0]
ode0 = np.array([-1., 0.]) # initial value for diffrax/odeint
theta = thetanp = None

# Time interval on which a solution is sought.
t_min = 0.
t_max = 10.

n_vars = 1                        # number of variables in the ODE
n_deriv = 4  # max number of derivatives
sigma = jnp.array([.1] * n_vars)  # IBM process scale factor

n_steps = 30                  # number of evaluations steps
dt = (t_max - t_min) / n_steps  # step size
tseq = jnp.linspace(t_min, t_max, n_steps+1)

# generate the Kalman parameters corresponding to the prior
prior_Q_block, prior_R_block = ibm_init(
    dt=dt,
    n_deriv=n_deriv,
    sigma=sigma
)

prior_Q, prior_R = indep_init(
    prior_weight=prior_Q_block,
    prior_var=prior_R_block
)

prior_Q = prior_Q[0] # remove the block
prior_R = prior_R[0] 


# Timings
n_loops = 500
n_times = 10
time_rd = rodeo_time(chkrebtii, n_loops, n_times) # rodeo
time_rdnb = rodeo_nb_time(chkrebtii_nb, n_loops, n_times) # rodeo non-block
time_ode = odeint_time(chkrebtii_ode, n_loops, n_times) # odeint
time_rax = diffrax_time(chkrebtii_rax, n_loops, n_times) # diffrax


with open("output/timings.txt", "w") as text_file:
    text_file.write("Chkrebtii----------------------------------\n")
    text_file.write("rodeo / LSODA = {}\n".format(time_ode/time_rd))
    text_file.write("rodeo / RKDP = {}\n".format(time_rax/time_rd))
    text_file.write("rodeo / non-blocking = {}\n".format(time_rdnb/time_rd))

# Fitz-Hugh Nagumo ---------------------------------------------------------------------------------------------------
# used by rodeo
def fitz(X, t, theta):
    "FitzHugh-Nagumo ODE."
    a, b, c = theta
    V, R = X[:,0]
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
def fitz_ode(X, t, theta):
    a, b, c = theta
    V, R = X
    return np.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

# used by diffrax
def fitz_rax(t, X, theta):
    a, b, c = theta
    V, R = X
    return jnp.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

n_vars = 2   # number of variables in the ODE
n_deriv = 3  # max number of derivatives
sigma = jnp.array([.1] * n_vars)  # IBM process scale factor

# initial value in various formats
ode0 = np.array([-1., 1.]) # initial value for diffrax/odeint
W_block, fitz_init_pad = first_order_pad(fitz, n_vars, n_deriv)
W = jsl.block_diag(*W_block)
theta = jnp.array([0.2, 0.2, 3])
thetanp = np.array(theta)
x0_block = fitz_init_pad(jnp.array(ode0), 0, theta=theta)  # initial value in rodeo format
x0 = x0_block.flatten()

# Time interval on which a solution is sought.
t_min = 0.
t_max = 40.

n_steps = 250                  # number of evaluations steps
dt = (t_max - t_min) / n_steps  # step size
tseq = jnp.linspace(t_min, t_max, n_steps+1)

# generate the Kalman parameters corresponding to the prior
prior_Q_block, prior_R_block = ibm_init(
    dt=dt,
    n_deriv=n_deriv,
    sigma=sigma
)

prior_Q, prior_R = indep_init(
    prior_weight=prior_Q_block,
    prior_var=prior_R_block
)

prior_Q = prior_Q[0] # remove the block
prior_R = prior_R[0] 

# Timings
n_loops = 500
n_times = 10
time_rd = rodeo_time(fitz, n_loops, n_times) # rodeo
time_rdnb = rodeo_nb_time(fitz_nb, n_loops, n_times) # rodeo non-block
time_ode = odeint_time(fitz_ode, n_loops, n_times) # odeint
time_rax = diffrax_time(fitz_rax, n_loops, n_times) # diffrax

with open("output/timings.txt", "a") as text_file:
    text_file.write("Fitz-Hugh----------------------------------\n")
    text_file.write("rodeo / LSODA = {}\n".format(time_ode/time_rd))
    text_file.write("rodeo / RKDP = {}\n".format(time_rax/time_rd))
    text_file.write("rodeo / non-blocking = {}\n".format(time_rdnb/time_rd))

# Hes1 ---------------------------------------------------------------------------------------------------------------
def hes1(X, t, theta):
    "Hes1 on the log-scale in rodeo format."
    P, M, H = jnp.exp(X[:, 0])
    a, b, c, d, e, f, g = theta
    logP = -a * H + b * M / P - c
    logM = -d + e / (1 + P * P) / M
    logH = -a * P + f / (1 + P * P) / H - g
    return jnp.array([[logP], [logM], [logH]])

def hes1_nb(X, t, theta):
    P, M, H = jnp.exp(X[::3])
    a, b, c, d, e, f, g = theta
    logP = -a * H + b * M / P - c
    logM = -d + e / (1 + P * P) / M
    logH = -a * P + f / (1 + P * P) / H - g
    return jnp.array([logP, logM, logH])

@njit
def hes1_ode(X, t, theta):
    P, M, H = np.exp(X)
    a, b, c, d, e, f, g = theta
    logP = -a * H + b * M / P - c
    logM = -d + e / (1 + P * P) / M
    logH = -a * P + f / (1 + P * P) / H - g
    return np.array([logP, logM, logH])

def hes1_rax(t, X, theta):
    P, M, H = jnp.exp(X)
    a, b, c, d, e, f, g = theta
    logP = -a * H + b * M / P - c
    logM = -d + e / (1 + P * P) / M
    logH = -a * P + f / (1 + P * P) / H - g
    return jnp.array([logP, logM, logH])


n_vars = 3                        # number of variables in the ODE
n_deriv = 3  # max number of derivatives
sigma = jnp.array([.1] * n_vars)  # IBM process scale factor

# initial value in various formats
ode0 = np.log(np.array([1.439, 2.037, 17.904])) # initial value for diffrax/odeint
W_block, hes1_init_pad = first_order_pad(hes1, n_vars, n_deriv)
W = jsl.block_diag(*W_block)
theta = jnp.array([0.022, 0.3, 0.031, 0.028, 0.5, 20, 0.3])  # ODE parameters
thetanp = np.array(theta)
x0_block = hes1_init_pad(jnp.array(ode0), 0, theta=theta)  # initial value in rodeo format
x0 = x0_block.flatten()

# Time interval on which a solution is sought.
t_min = 0.
t_max = 240.

n_steps = 120                  # number of evaluations steps
dt = (t_max - t_min) / n_steps  # step size
tseq = jnp.linspace(t_min, t_max, n_steps+1)

# generate the Kalman parameters corresponding to the prior
prior_Q_block, prior_R_block = ibm_init(
    dt=dt,
    n_deriv=n_deriv,
    sigma=sigma
)

prior_Q, prior_R = indep_init(
    prior_weight=prior_Q_block,
    prior_var=prior_R_block
)

prior_Q = prior_Q[0] # remove the block
prior_R = prior_R[0] 

# Timings
n_loops = 500
n_times = 10
time_rd = rodeo_time(hes1, n_loops, n_times) # rodeo
time_rdnb = rodeo_nb_time(hes1_nb, n_loops, n_times) # rodeo non-block
time_ode = odeint_time(hes1_ode, n_loops, n_times) # odeint
time_rax = diffrax_time(hes1_rax, n_loops, n_times) # diffrax

with open("output/timings.txt", "a") as text_file:
    text_file.write("Hes1----------------------------------\n")
    text_file.write("rodeo / LSODA = {}\n".format(time_ode/time_rd))
    text_file.write("rodeo / RKDP = {}\n".format(time_rax/time_rd))
    text_file.write("rodeo / non-blocking = {}\n".format(time_rdnb/time_rd))

# SEIRAH -------------------------------------------------------------------------------------------------------------
def seirah(X, t, theta):
    "SEIRAH ODE in rodeo format."
    S, E, I, R, A, H = X[:, 0]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b * S * (I + alpha * A) / N
    dE = b * S * (I + alpha * A) / N - E / D_e
    dI = r * E / D_e - I / D_q - I / D_I
    dR = (I + A) / D_I + H / D_h
    dA = (1 - r) * E / D_e - A / D_I
    dH = I / D_q - H / D_h
    return jnp.array([[dS], [dE], [dI], [dR], [dA], [dH]])

def seirah_nb(X, t, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X[::3]
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b * S * (I + alpha * A) / N
    dE = b * S * (I + alpha * A) / N - E / D_e
    dI = r * E / D_e - I / D_q - I / D_I
    dR = (I + A) / D_I + H / D_h
    dA = (1 - r) * E / D_e - A / D_I
    dH = I / D_q - H / D_h
    return jnp.array([dS, dE, dI, dR, dA, dH])

@njit
def seirah_ode(X, t, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b*S*(I + alpha*A)/N
    dE = b*S*(I + alpha*A)/N - E/D_e
    dI = r*E/D_e - I/D_q - I/D_I
    dR = (I + A)/D_I + H/D_h
    dA = (1-r)*E/D_e - A/D_I
    dH = I/D_q - H/D_h
    return np.array([dS, dE, dI, dR, dA, dH])

def seirah_rax(t, X, theta):
    "SEIRAH ODE function"
    S, E, I, R, A, H = X
    N = S + E + I + R + A + H
    b, r, alpha, D_e, D_I, D_q = theta
    D_h = 30
    dS = -b*S*(I + alpha*A)/N
    dE = b*S*(I + alpha*A)/N - E/D_e
    dI = r*E/D_e - I/D_q - I/D_I
    dR = (I + A)/D_I + H/D_h
    dA = (1-r)*E/D_e - A/D_I
    dH = I/D_q - H/D_h
    return jnp.array([dS, dE, dI, dR, dA, dH])

def seirah_init(x0, theta):
    "SEIRAH initial values in rodeo format."
    x0 = x0[:, None]
    return jnp.hstack([
        x0,
        seirah(X=x0, t=0., theta=theta),
        jnp.zeros_like(x0)
    ])

n_vars = 6                        # number of variables in the ODE
n_deriv = 3  # max number of derivatives
sigma = jnp.array([.1] * n_vars)  # IBM process scale factor

# initial value in various formats
ode0 = np.array([63804435., 15492., 21752., 0., 618013., 93583.]) # initial value for diffrax/odeint
W_block, seirah_init_pad = first_order_pad(seirah, n_vars, n_deriv)
W = jsl.block_diag(*W_block)
theta = jnp.array([2.23, 0.034, 0.55, 5.1, 2.3, 1.13])  # ODE parameters
thetanp = np.array(theta)
x0_block = seirah_init_pad(jnp.array(ode0), 0, theta=theta) # initial value in rodeo format
x0 = x0_block.flatten()

# Time interval on which a solution is sought.
t_min = 0.
t_max = 60.

n_steps = 80                  # number of evaluations steps
dt = (t_max - t_min) / n_steps  # step size
tseq = jnp.linspace(t_min, t_max, n_steps+1)

# generate the Kalman parameters corresponding to the prior
prior_Q_block, prior_R_block = ibm_init(
    dt=dt,
    n_deriv=n_deriv,
    sigma=sigma
)

prior_Q, prior_R = indep_init(
    prior_weight=prior_Q_block,
    prior_var=prior_R_block
)

prior_Q = prior_Q[0] # remove the block
prior_R = prior_R[0] 


# Timings
n_loops = 500
n_times = 10
time_rd = rodeo_time(seirah, n_loops, n_times) # rodeo
time_rdnb = rodeo_nb_time(seirah_nb, n_loops, n_times) # rodeo non-block
time_ode = odeint_time(seirah_ode, n_loops, n_times) # odeint
time_rax = diffrax_time(seirah_rax, n_loops, n_times) # diffrax

with open("output/timings.txt", "a") as text_file:
    text_file.write("SEIRAH----------------------------------\n")
    text_file.write("rodeo / LSODA = {}\n".format(time_ode/time_rd))
    text_file.write("rodeo / RKDP = {}\n".format(time_rax/time_rd))
    text_file.write("rodeo / non-blocking = {}\n".format(time_rdnb/time_rd))

