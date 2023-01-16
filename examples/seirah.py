import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config

from inference.seirah_inference import seirah_inference
from inference.theta_plot import *
from rodeo.ibm import ibm_init
from rodeo.ode import *
config.update("jax_enable_x64", True)

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

def seirah_example(load_calcs=False):
    "Perform parameter inference using the SEIRAH model."
    n_vars = 6  # number of system variables
    # number of continuous derivatives per variable
    n_deriv = jnp.array([3] * n_vars)
    sigma = jnp.array([.1] * n_vars)  # IBM process scale factor per variable
    
    # time interval on which solution is sought
    tmin = 0.
    tmax = 60.

    # Initial x0 for odeint
    ode0 = np.array([63884630, 15492, 21752, 0, 618013, 13388])

    # ODE LHS matrix
    W_mat = np.zeros((n_vars, 1, 3))
    W_mat[:, :, 1] = 1
    W = jnp.array(W_mat)

    # logprior parameters
    theta_true = jnp.array([2.23, 0.034, 0.55, 5.1, 2.3, 1.13]) # True theta
    n_phi = 6
    phi_mean = jnp.zeros(n_phi)
    phi_sd = jnp.log(10)*jnp.ones(n_phi) 
    n_theta = len(theta_true)
    
    dt_obs = 1.0  # Time between observations

    # Number of samples to draw from posterior
    n_samples = 100000

    # Initialize inference class and simulate observed data
    key = jax.random.PRNGKey(0)
    mask = jnp.array([1,2])
    n_theta = len(theta_true)

    inf = seirah_inference(key, seirah, W, tmin, tmax, phi_mean, phi_sd, mask, n_theta)
    Y_t, X_t = inf.simulate(ode0, theta_true)
    
    # Initial value, x0, for the IVP
    x0 = jnp.array([[63884630.], [-1.], [-1.], [0.], [618013.], [13388.]]) # -1 for missing values

    n_res_list = np.array([10, 20, 50, 100])
    phi_init = jnp.append(jnp.log(theta_true), jnp.log(jnp.array([15492., 21752.])))
    if load_calcs:
        theta_kalman = np.load('saves/seirah_theta_kalman.npy')
        theta_diffrax = np.load('saves/seirah_theta_diffrax.npy')
    else:
        
        # Parameter inference using Kalman solver
        theta_kalman = np.zeros((len(n_res_list), n_samples, len(phi_init)))
        for i in range(len(n_res_list)):
            prior_pars = ibm_init(1/n_res_list[i], n_deriv, sigma)
            n_steps = int((tmax-tmin)*n_res_list[i])
            print(n_steps)
            inf.n_steps = n_steps
            inf.n_res = n_res_list[i]
            inf.prior_pars = prior_pars
            phi_hat, phi_var = inf.phi_fit(phi_init, x0, inf.kalman_nlpost)
            theta_kalman[i] = inf.phi_sample(phi_hat, phi_var, n_samples)
            theta_kalman[i] = np.exp(theta_kalman[i])
        # np.save('saves/seirah_theta_kalman.npy', theta_kalman)
        
        # Parameter inference using diffrax
        inf.diff_dt0 = 0.1
        phi_hat, phi_var =  inf.phi_fit(phi_init, x0, inf.diffrax_nlpost)
        theta_diffrax = inf.phi_sample(phi_hat, phi_var, n_samples)
        theta_diffrax = np.exp(theta_diffrax)
        # np.save('saves/seirah_theta_diffrax.npy', theta_diffrax)

    plt.rcParams.update({'font.size': 20})
    var_names = ["b", "r", r"$\alpha$", "$D_e$", "$D_I$", "$D_q$", "$E(0)}$", "$I(0)}$"]
    clip = [(0, 8), None, (0,2), None, None, None, (0, 30000), None]
    param_true = np.append(theta_true, np.array([15492, 21752]))
    figure = theta_plotsingle(theta_kalman[:4], theta_diffrax, param_true, 1/n_res_list, var_names, clip=clip, rows=2)
    figure.savefig('figures/seirahfigure.pdf')
    plt.show()
    return

