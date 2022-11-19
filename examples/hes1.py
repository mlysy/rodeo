import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.config import config

from inference.hes1inf import hes1inf as inference
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

def hes1pad(X_t, t, theta):
    P, M, H = jnp.exp(X_t)
    a, b, c, d, e, f, g = theta
    x1 = -a*H + b*M/P - c
    x2 = -d + e/(1+P*P)/M
    x3 = -a*P + f/(1+P*P)/H - g
    return jnp.array([[X_t[0], x1, 0],
                      [X_t[1], x2, 0],
                      [X_t[2], x3, 0]])


def hes1_example(load_calcs=False):
    "Perform parameter inference using the Hes1 function."
    # problem setup and intialization
    n_deriv = 1 # number of derivatives in IVP
    n_obs = 3 # number of observations
    n_deriv_prior =  3 # number of derivatives in IBM prior

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    tmin = 0.
    tmax = 240.

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    sigma = jnp.array([.001]*n_obs)
    n_order = jnp.array([n_deriv_prior]*n_obs)

    # Initial x0 for odeint
    x0 = np.log(np.array([1.439, 2.037, 17.904]))

    # pad the inputs
    W_mat = np.zeros((n_obs, 1, n_deriv_prior))
    W_mat[:, :, 1] = 1
    W = jnp.array(W_mat)

    # logprior parameters
    theta_true = np.array([0.022, 0.3, 0.031, 0.028, 0.5, 20, 0.3]) # True theta
    n_theta = len(theta_true)
    phi_mean = np.zeros(n_theta)
    phi_sd = np.log(10)*np.ones(n_theta) 

    # Observation noise
    gamma = 0.15

    # Number of samples to draw from posterior
    n_samples = 100000

    # Initialize inference class and simulate observed data
    key = jax.random.PRNGKey(0)
    inf = inference(key, tmin, tmax, hes1)
    inf.funpad = hes1pad
    tseq = np.linspace(tmin, tmax, 33)
    Y_t, X_t = inf.simulate(x0, theta_true, gamma, tseq)
    
    # # exp observations for plot
    # Y_exp = np.exp(Y_t) 
    # X_exp = np.exp(X_t)

    # plt.rcParams.update({'font.size': 20})
    # fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    # axs[0].plot(tseq[::2], X_exp[:17], label = 'X_t')
    # axs[0].scatter(tseq[::2], Y_exp[:17], label = 'Y_t', color='orange')
    # axs[0].set_title("$P^{(0)}_t$")
    # axs[1].plot(tseq[1::2], X_exp[17:], label = 'X_t')
    # axs[1].scatter(tseq[1::2], Y_exp[17:], label = 'Y_t', color='orange')
    # axs[1].set_title("$M^{(0)}_t$")
    # axs[1].legend(loc='upper left', bbox_to_anchor=[1, 1])
    # # fig.savefig('figures/hes1sim.pdf')
    
    dtlst = np.array([2.0, 1.5, 1.0, 0.5])
    obs_t = 7.5
    if load_calcs:
        theta_kalman = np.load('saves/hes1_theta_kalman.npy')
        theta_diffrax = np.load('saves/hes1_theta_diffrax.npy')
    else:

        phi_init = jnp.append(np.log(theta_true), x0)
        # Parameter inference using Kalman solver
        theta_kalman = np.zeros((len(dtlst), n_samples, n_theta+3))
        for i in range(len(dtlst)):
            kinit = ibm_init(dtlst[i], n_order, sigma)
            n_eval = int((tmax-tmin)/dtlst[i])
            inf.n_eval = n_eval
            inf.kinit = kinit
            inf.W = W
            phi_hat, phi_var = inf.phi_fit(Y_t, np.array([None, None, None]), dtlst[i], obs_t, phi_mean, phi_sd, inf.kalman_nlpost,
                                           gamma, phi_init = phi_init)
            theta_kalman[i] = inf.phi_sample(phi_hat, phi_var, n_samples)
            theta_kalman[i] = np.exp(theta_kalman[i])
        # np.save('saves/hes1_theta_kalman.npy', theta_kalman)

        # Parameter inference using diffrax solver
        phi_hat, phi_var = inf.phi_fit(Y_t, np.array([None, None, None]), 1.5, obs_t, phi_mean, phi_sd, inf.diffrax_nlpost,
                                       gamma, phi_init = phi_init)
        theta_diffrax = inf.phi_sample(phi_hat, phi_var, n_samples)
        theta_diffrax = np.exp(theta_diffrax)
        #np.save('saves/hes1_theta_diffrax.npy', theta_diffrax)
    # Produces the graph in Figure 3
    plt.rcParams.update({'font.size': 20})
    var_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', r"$P(0)$", r"$M(0)$", r"$H(0)$"]
    clip = [(0.00, 0.15), None, None, None, None, (0, 30), (0, 1), None, None, (0,30)]
    param_true = np.append(theta_true, np.exp(x0))
    figure = inf.theta_plotsingle(theta_kalman, theta_diffrax, param_true, dtlst, var_names, clip=clip, rows=2)
    figure.savefig('figures/hes1figure.pdf')
    plt.show()
    return

if __name__ == '__main__':
    hes1_example(False)
    