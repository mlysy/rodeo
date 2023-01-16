import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config

from inference.hes1_inference import hes1_inference
from inference.theta_plot import *
from rodeo.ibm import ibm_init
from rodeo.ode import *
config.update("jax_enable_x64", True)

def hes1(X_t, t, theta):
    "Hes1 model on the log-scale"
    P, M, H = jnp.exp(X_t[:, 0])
    a, b, c, d, e, f, g = theta
    
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return jnp.array([[x1], [x2], [x3]])


def hes1_example(load_calcs=False):
    "Perform parameter inference using the Hes1 function."
    n_vars = 3  # number of system variables
    # number of continuous derivatives per variable
    n_deriv = jnp.array([3] * n_vars)
    sigma = jnp.array([.001] * n_vars)  # IBM process scale factor per variable
    
    # time interval on which solution is sought
    tmin = 0.
    tmax = 240.

    # Initial x0 for odeint
    ode0 = np.log(np.array([1.439, 2.037, 17.904]))

    # ODE LHS matrix
    W_mat = np.zeros((n_vars, 1, 3))
    W_mat[:, :, 1] = 1
    W = jnp.array(W_mat)

    # logprior parameters
    theta_true = jnp.array([0.022, 0.3, 0.031, 0.028, 0.5, 20, 0.3]) # True theta
    n_phi = 7
    phi_mean = jnp.zeros(n_phi)
    phi_sd = jnp.log(10)*jnp.ones(n_phi) 
    n_theta = len(theta_true)
    
    noise_sigma = 0.15  # Standard deviation in noise model
    dt_obs = 7.5  # Time between observations

    # Number of samples to draw from posterior
    n_samples = 100000

    # Initialize inference class and simulate observed data
    key = jax.random.PRNGKey(0)
    mask = range(3)
    n_theta = len(theta_true)

    inf = hes1_inference(key, hes1, W, tmin, tmax, phi_mean, phi_sd, mask, noise_sigma, n_theta)
    Y_t, X_t = inf.simulate(ode0, theta_true)
    # perform inference for various step_sizes
    n_res_list = np.array([4, 5, 6, 10])
    dt_list = dt_obs/n_res_list
    phi_init = np.append(np.log(theta_true), ode0)
    if load_calcs:
        theta_kalman = np.load('saves/hes1_theta_kalman.npy')
        theta_diffrax = np.load('saves/hes1_theta_diffrax.npy')
    else:
        # Parameter inference using Kalman solver
        theta_kalman = np.zeros((len(n_res_list), n_samples, len(phi_init)))
        for i in range(len(n_res_list)):
            prior_pars = ibm_init(dt_list[i], n_deriv, sigma)
            n_steps = int((tmax-tmin)/dt_list[i])
            print(n_steps)
            inf.n_steps = n_steps
            inf.n_res = n_res_list[i]
            inf.prior_pars = prior_pars
            phi_hat, phi_var = inf.phi_fit(phi_init, jnp.zeros((3,1)), inf.kalman_nlpost)
            theta_kalman[i] = inf.phi_sample(phi_hat, phi_var, n_samples)
            theta_kalman[i] = np.exp(theta_kalman[i])
        # np.save('saves/hes1_theta_kalman.npy', theta_kalman)

        # Parameter inference using diffrax solver
        inf.diff_dt0 = 1.5
        phi_hat, phi_var =  inf.phi_fit(phi_init, jnp.zeros((3,1)), inf.diffrax_nlpost)
        theta_diffrax = inf.phi_sample(phi_hat, phi_var, n_samples)
        theta_diffrax = np.exp(theta_diffrax)
        # np.save('saves/fitz_theta_diffrax.npy', theta_diffrax)
    
    # Produces the graph in Figure 3
    plt.rcParams.update({'font.size': 20})
    var_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', r"$P(0)$", r"$M(0)$", r"$H(0)$"]
    clip = [(0.00, 0.15), None, None, None, None, (0, 30), (0, 1), None, None, (0,30)]
    param_true = np.append(theta_true, np.exp(ode0))
    figure = theta_plotsingle(theta_kalman, theta_diffrax, param_true, dt_list, var_names, clip=clip, rows=2)
    figure.savefig('figures/hes1figure.pdf')
    plt.show()
    return

if __name__ == '__main__':
    hes1_example(False)
    