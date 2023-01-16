import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config

from inference.fitz_inference import fitz_inference
from inference.theta_plot import *
from rodeo.ibm import ibm_init
from rodeo.ode import *
config.update("jax_enable_x64", True)

def fitz(X_t, t, theta):
    "Fitz ODE written for jax"
    a, b, c = theta
    V, R = X_t[:, 0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                      [-1/c*(V - a + b*R)]])

def fitz_example(load_calcs=False):
    "Perform parameter inference using the FitzHugh-Nagumo function."
    n_vars = 2  # number of system variables
    # number of continuous derivatives per variable
    n_deriv = jnp.array([3] * n_vars)
    sigma = jnp.array([.1] * n_vars)  # IBM process scale factor per variable
    
    # time interval on which solution is sought
    tmin = 0.
    tmax = 40.

    W = jnp.array([[[0., 1., 0.]], [[0., 1., 0.]]])  # ODE LHS matrix
    x0 = jnp.array([[-1., 1., 0.], [1., 1/3, 0.]])  # IVP initial value
    
    # logprior parameters
    theta_true = jnp.array([0.2, 0.2, 3]) # True theta
    n_phi = 5
    phi_mean = jnp.zeros(n_phi)
    phi_sd = jnp.log(10)*jnp.ones(n_phi) 

    noise_sigma = 0.2  # Standard deviation in noise model
    dt_obs = 1.0  # Time between observations

    # Number of samples to draw from posterior
    n_samples = 100000

    # Initialize inference class and simulate observed data
    key = jax.random.PRNGKey(0)
    mask = range(2)
    n_theta = len(theta_true)
    tseq = np.linspace(tmin, tmax, 41)
    ode0 = x0[:, 0]
    
    inf = fitz_inference(key, fitz, W, tmin, tmax, phi_mean, phi_sd, mask, noise_sigma, n_theta)
    Y_t, X_t = inf.simulate(ode0, theta_true)
    # np.save('saves/fitz_Y_t.npy', Y_t)
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    axs[0].plot(tseq, X_t[:,0], label = 'X_t')
    axs[0].scatter(tseq, Y_t[:,0], label = 'Y_t', color='orange')
    axs[0].set_title("$V(t)$")
    axs[1].plot(tseq, X_t[:,1], label = 'X_t')
    axs[1].scatter(tseq, Y_t[:,1], label = 'Y_t', color='orange')
    axs[1].set_title("$R(t)$")
    axs[1].legend(loc='upper left', bbox_to_anchor=[1, 1])
    # fig.savefig('figures/fitzsim.pdf')
    
    n_res_list = np.array([10, 20, 50, 100])
    phi_init = jnp.append(jnp.log(theta_true), ode0)
    if load_calcs:
        theta_euler = np.load('saves/fitz_theta_euler.npy')
        theta_kalman = np.load('saves/fitz_theta_kalman.npy')
        theta_diffrax = np.load('saves/fitz_theta_diffrax.npy')
    else:
        # Parameter inference using Euler's approximation
        theta_euler = np.zeros((len(n_res_list), n_samples, n_phi))
        for i in range(len(n_res_list)):
            n_steps = int((tmax-tmin)*n_res_list[i])
            inf.n_steps = n_steps
            inf.n_res = n_res_list[i]
            phi_hat, phi_var = inf.phi_fit(phi_init, jnp.zeros((2,1)), inf.euler_nlpost)
            theta_euler[i] = inf.phi_sample(phi_hat, phi_var, n_samples)
            theta_euler[i, :, :n_theta] = np.exp(theta_euler[i, :, :n_theta])
            
        # np.save('saves/fitz_theta_euler.npy', theta_euler)
        
        # Parameter inference using Kalman solver
        theta_kalman = np.zeros((len(n_res_list), n_samples, n_phi))
        for i in range(len(n_res_list)):
            prior_pars = ibm_init(1/n_res_list[i], n_deriv, sigma)
            n_steps = int((tmax-tmin)*n_res_list[i])
            inf.n_steps = n_steps
            inf.n_res = n_res_list[i]
            inf.prior_pars = prior_pars
            phi_hat, phi_var = inf.phi_fit(phi_init, jnp.zeros((2,1)), inf.kalman_nlpost)
            theta_kalman[i] = inf.phi_sample(phi_hat, phi_var, n_samples)
            theta_kalman[i, :, :n_theta] = np.exp(theta_kalman[i, :, :n_theta])
        # np.save('saves/fitz_theta_kalman.npy', theta_kalman)

        # Parameter inference using diffrax
        inf.diff_dt0 = dt_obs
        phi_hat, phi_var =  inf.phi_fit(phi_init, jnp.zeros((2,1)), inf.diffrax_nlpost)
        theta_diffrax = inf.phi_sample(phi_hat, phi_var, n_samples)
        theta_diffrax[:, :n_theta] = np.exp(theta_diffrax[:, :n_theta])
        # np.save('saves/fitz_theta_diffrax.npy', theta_diffrax)
        
    # Produces the graph in Figure 3
    plt.rcParams.update({'font.size': 20})
    var_names = ['a', 'b', 'c', r"$V(0)$", r"$R(0)$"]
    param_true = np.append(theta_true, np.array([-1, 1]))
    figure = theta_plot(theta_euler, theta_kalman, theta_diffrax, param_true, 1/n_res_list, var_names, clip=[None, (0, 0.5), None, None, None], rows=1)
    figure.savefig('figures/fitzfigure.pdf')
    plt.show()
    return

if __name__ == '__main__':
    fitz_example(False)
