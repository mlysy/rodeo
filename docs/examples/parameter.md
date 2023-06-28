---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Parameter Inference
In this notebook, we demonstrate the steps to conduct parameter inference using various ODE solvers in **rodeo**.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from scipy.integrate import odeint
from jax import jacfwd, jacrev
from jaxopt import ScipyMinimize

from rodeo.ibm import ibm_init
from rodeo.ode import interrogate_tronarp, interrogate_chkrebtii
import rodeo.ode as ro
import rodeo.fenrir as rf
import rodeo.dalton as rd

import matplotlib.pyplot as plt
import seaborn as sns
from jax.config import config
config.update("jax_enable_x64", True)
```

## Walkthrough

We will use the FitzHugh-Nagumo model as the example here which is a two-state ODE on $\xx(t) = (V(t), R(t))$,

\begin{align*}
\frac{dV(t)}{dt} &= c(V(t) - \frac{V(t)^3}{3} + R(t)), \\
\frac{dR(t)}{dt} &= -\frac{(V(t) - a - bR(t))}{c}. \\
\end{align*}

The model parameters are $\tth = (a,b,c,V(0),R(0))$, with $a,b,c > 0$ which are to be learned from the measurement model

\begin{align*}
\YY_i \sim \N(\xx(t_i), \phi^2 \Id_{2 \times 2})
\end{align*}

where $t_i = i$ and $i=0,1,\ldots 40$ and $\phi^2 = 0.005$. We will first simulate some noisy data using an highly accurate ODE solver (`odeint`).

```{code-cell} ipython3
def fitz0(X_t, t, theta):
    a, b, c = theta
    V, R = X_t 
    return np.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

# it is assumed that the solution is sought on the interval [tmin, tmax].
tmin = 0.
tmax = 40.
theta = np.array([0.2, 0.2, 3])

# Initial x0 for odeint
ode0 = np.array([-1., 1.])

# observations
n_obs = 40
tseq = np.linspace(tmin, tmax, n_obs+1)
exact = odeint(fitz0, ode0, tseq, args=(theta,))
noise_sd = 0.2
et = np.random.default_rng(0).normal(loc=0.0, scale=1, size=exact.shape)
Yt = exact + noise_sd*et
```

```{code-cell} ipython3
# plot one graph
plt.rcParams.update({'font.size': 30})
fig1, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].plot(tseq, exact[:,0], label = 'True', linewidth=4)
axs[0].scatter(tseq, Yt[:,0], label = 'Obs', color='orange', s=200, zorder=2)
axs[0].set_title("$V(t)$")
axs[1].plot(tseq, exact[:,1], label = 'True', linewidth=4)
axs[1].scatter(tseq, Yt[:,1], label = 'Obs', color='orange', s=200, zorder=2)
axs[1].set_title("$R(t)$")
axs[1].legend(loc=1)
fig1.tight_layout()
```

We define the Gaussian Markov process prior using the IBM. Other choices are possible, however, we have made this simple prior available in the library. The rest of the inputs are the standard inputs to `rodeo` as described in the Introduction notebook. In addition, we define `n_res` as the resolution of the solution based on the observations. For example, `n_res=10` in this ODE would give `n_step = 10 * 40 = 400`.

```{code-cell} ipython3
def fitz(X_t, t, theta):
    "Fitz ODE written for jax"
    a, b, c = theta
    V, R = X_t[:, 0]
    return jnp.array([[c*(V - V*V*V/3 + R)],
                      [-1/c*(V - a + b*R)]])
# ode parameter 
theta = jnp.array([0.2, 0.2, 3])

# problem setup and intialization
n_deriv = 1  # Total state; q
n_var = 2  # Total variables
n_deriv_prior = 3 # p

# tuning parameter in the IBM process
sigma = jnp.array([.1]*n_var) 

# block definition for W, and x0
W = jnp.array([[[0., 1., 0.]], [[0., 1., 0.]]])  # ODE LHS matrix
x0 = jnp.array([[-1., 1., 0.], [1., 1/3, 0.]])

# Get parameters needed to run the solver
n_res = 10
n_steps = n_obs*n_res
dt = (tmax-tmin)/n_steps
n_order = jnp.array([n_deriv_prior]*n_var)
prior_pars = ibm_init(dt, n_order, sigma)

# prng key
key = jax.random.PRNGKey(0)
```

We proceed with a Bayesian approach by postulating a prior distribution $\pi(\tth)$ which combined with the likelihood gives the posterior
\begin{equation}\label{eq:likelihood}
    p(\tth \mid \YY_{0:M}) \propto \pi(\tth) \times p(\YY_{0:M} \mid \ZZ_{0:N} = \bz, \tth)
\end{equation}
where $p(\YY_{0:M} \mid \ZZ_{0:N} = \bz, \tth)$ is computed with different methods.
Parameter inference is then accomplished by way of a Laplace approximation, for which we have
\begin{equation*}
    \tth \mid \YY_{0:M} \approx \N(\hat \tth, \hat \VV_{\tth}),
\end{equation*}
where $\hat \tth = \argmax_{\tth} \log p(\tth \mid \YY_{0:M})$ and $\hat \VV_{\tth} = -\big[\frac{\partial^2}{\partial \tth \partial \tth'} \log p(\hat \tth \mid \YY_{0:M})\big]^{-1}$. For the prior, we assume independent $\N(0, 10^2)$ priors on $\log a, \log b, \log c$ and $V(0), R(0)$.

```{code-cell} ipython3
# logprior parameters
theta_true = jnp.array([0.2, 0.2, 3]) # True theta
n_phi = 5
phi_mean = jnp.zeros(n_phi)
phi_sd = jnp.log(10)*jnp.ones(n_phi) 
n_theta = 3
n_samples = 100000
```

### Basic Likelihood

To start, we use a standard probabilistic ODE solver, `rodeo`, to construct a likelihood. An appropriate estimation to $p(\YY_{0:M} \mid \ZZ_{0:N} = \bz, \tth)$ in (1) is $\prod_{i=0}^M p(\YY_i \mid \XX(t_i) = \mmu_{n(i)|N}, \tth)$ where $n(i)$ maps the corresponding time points of the solver to the data. 

We define the prior and posterior functions necessary to conduct parameter inference. The key function to focus is `basic_nlpost` which takes in `phi = (log a, log b, log c, V(0), R(0))`. It uses the ODE and the $V(0), R(0)$ to compute the  $dV(0), dR(0)$ required for `rodeo` and then zero pad it an extra dimension which helps with accuracy as explained in the Introduction notebook. The other three parameters $\log a, \log b, \log c$ needs to be exponentiated first because `rodeo` assumes they are on the regular scale. The function `phi_fit` essentially computes the Laplace approximation detailed above. We use the **jaxopt** library as it supports optimization using **jax**. We choose `Newton-CG` as our optimization algorithm but there are many other possible choices. Consult the **jaxopt** documentation if you are interested in this library.

```{code-cell} ipython3
def logprior(x, mean, sd):
    r"Calculate the loglikelihood of the normal distribution."
    return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

def basic_nlpost(phi):
    r"Compute the negative loglikihood of :math:`Y_t` using rodeo."
    x0 = phi[n_theta:].reshape((2,1))
    theta = jnp.exp(phi[:n_theta])
    v0 = fitz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    Xt, _ = ro.solve_mv(
        key=key,
        fun=fitz,
        W=W,
        x0=x0,
        theta=theta,
        tmin=tmin,
        tmax=tmax,
        interrogate=interrogate_tronarp,
        n_steps=n_steps,
        **prior_pars
    )
    # compute the loglikelihood and the log-prior
    loglik = jnp.sum(jsp.stats.norm.logpdf(
        x=Yt,
        loc=Xt[::n_res, :, 0],  # thin solver output
        scale=noise_sd
    ))
    logprior = jnp.sum(jsp.stats.norm.logpdf(
        x=phi,
        loc=phi_mean,
        scale=phi_sd
    ))
    
    return -(loglik + logprior)

def phi_fit(phi_init):
    r"""Compute the optimized :math:`\log{\theta}` and its variance given 
        :math:`Y_t`."""
    n_phi = len(phi_init)
    hes = jacfwd(jacrev(basic_nlpost))
    solver = ScipyMinimize(method="Newton-CG", fun = basic_nlpost)
    opt_res = solver.run(phi_init)
    phi_hat = opt_res.params
    phi_fisher = hes(phi_hat)
    phi_var = jsp.linalg.solve(phi_fisher, jnp.eye(n_phi))
    return phi_hat, phi_var

def phi_sample(phi_hat, phi_var, n_samples):
    r"""Simulate :math:`\theta` given the :math:`\log{\hat{\theta}}` 
        and its variance."""
    phi = np.random.default_rng(12345).multivariate_normal(phi_hat, phi_var, n_samples)
    return phi
```

```{code-cell} ipython3
# optimization process
phi_init = jnp.append(jnp.log(theta_true), ode0)
phi_hat, phi_var = phi_fit(phi_init)
basic_post = phi_sample(phi_hat, phi_var, n_samples)
basic_post[:, :n_theta] = np.exp(basic_post[:, :n_theta])
```

### Chkrebtii MCMC

Another method to estimate the parameter posteriors of $\tth$ is given by [Chkrebtii et al (2016)](https://projecteuclid.org/euclid.ba/1473276259) using MCMC. First, $\tth_0$ is initialized from a given prior $\pi(\tth)$. Next, a sample solution, $\xx_{0:N}$ dependent on $\tth_0$ is computed from `rodeo` with `interrogate_chkrebtii`. At each sampling step, $\tth' \sim q(\tth_{i-1})$ is sampled from the proposal distribution and is used to compute a new sample solution, $\xx_{0:N}'$. Finally, a rejection ratio is used to decide if $\tth_i = \tth'$ is updated or $\tth_i = \tth_{i-1}$ is kept. 

The structure of this method is different than the other solvers presented here. That is, the MCMC method uses a base class to implement the skeleton of the algorithm with a few functions that need to be implemented by the user. First the `logprior` and `loglik` methods define the log-prior and loglikelihood respectively. Next, the `solve` method allows users to define how the algorithms use the inputs to solve the ODE. We provide a simple one in the base class, however, for this example, the initial values are part of parameters so a custom `solve` is required. Finally, `mcmc_sample` is a simple way to sample from the MCMC algorithm. The variable `param` is used to define the proposal distribution. Please see the documentation for more details.

```{code-cell} ipython3
import rodeo.oc_mcmc as rc

class fitz_ocmcmc(rc.oc_mcmc):

    def logprior(self, phi):
        r"Calculate the loglikelihood of the prior."
        return jnp.sum(jsp.stats.norm.logpdf(x=phi, loc=phi_mean, scale=phi_sd))

    def loglik(self, Xt):
        r"Calculate the loglikelihood of the observations."
        return jnp.sum(jsp.stats.norm.logpdf(x=self.y_obs, loc=Xt, scale=noise_sd))

    def solve(self, key, phi):
        r"Solve the ODE given the theta"
        x0 = jnp.expand_dims(phi[n_theta:],-1)
        theta = jnp.exp(phi[:n_theta])
        v0 = self.fun(x0, 0, theta)
        x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
        Xt = ro.solve_sim(key, self.fun, self.W, x0, theta, self.tmin, self.tmax, self.n_steps, 
                          **self.prior_pars, interrogate=interrogate_chkrebtii)
        Xt = Xt[::self.n_res, :, 0]
        return Xt
    
    def mcmc_sample(self, key, phi_init, n_samples):
        param = jnp.diag(jnp.array([0.0001, 0.01, 0.0001, 0.0001, 0.0001]))
        key, subkey = jax.random.split(key)
        initial_state = self.init(subkey, phi_init)
        def one_step(state, key):
            state, sample = self.step(key, state, param)
            return state, sample

        keys = jax.jax.random.split(key, n_samples)
        _, samples = jax.lax.scan(one_step, initial_state, keys)
        return samples['theta']
```

```{code-cell} ipython3
fitz_ch = fitz_ocmcmc(fitz, W, None, tmin, tmax, n_steps, n_res, prior_pars, Yt)
mcmc_post = np.array(fitz_ch.mcmc_sample(key, phi_init, n_samples))
mcmc_post[:, :n_theta] = np.exp(mcmc_post[:, :n_theta])
```

### Fenrir

`fenrir` is a method developed by [Tronarp et al (2022)](https://proceedings.mlr.press/v162/tronarp22a.html) which uses the data itself in the solution process. `fenrir` begins by using the data-free forward pass of `rodeo` to estimate $p(\XX_{0:N} \mid \ZZ_{0:N} = \bz, \tth)$. This model can be simulated from via a (non-homogeneous) Markov chain going backwards in time,
\begin{equation}\label{eq:fenrirback}
    \begin{aligned}
        \XX_N & \sim \N(\bb_N, \CC_N) \\
        \XX_n \mid \XX_{n+1} & \sim \N(\AA_n \XX_n + \bb_n, \CC_n),
    \end{aligned}
\end{equation}
where the coefficients $\AA_{0:N-1}$, $\bb_{0:N}$, and $\CC_{0:N}$ can be derived using the Kalman filtering and smoothing recursions. Next, `fenrir` assumes that Gaussian observations are added to the model, from which $p(\YY_{0:M} \mid \ZZ_{0:N} = \bz, \tth)$ is computed using a Kalman filter on the backward pass of (2).

To use `fenrir`, we first need to define the specifications. `fenrir` expects observations to be of the form
\begin{equation*}
\YY_i \sim \N(\DD \XX_i + \cc, \OOm).
\end{equation*}
This translates to the following set of definitions for this 2-state ODE.

```{code-cell} ipython3
# format observations to be taken by fenrir
y_obs = jnp.expand_dims(Yt, -1) 
trans_obs = jnp.array([[[1., 0., 0.]], [[1., 0., 0.]]]) 
mean_obs = jnp.zeros((2, 1))
var_obs = noise_sd**2*jnp.array([[[1.]],[[1.]]])
```

The way to construct a likelihood estimation is very similar to the basic method. The one difference is that since `fenrir` gives a direct approximation to $p(\YY_{0:M} \mid \ZZ_{0:N} = \bz, \tth)$, the function `fenrir_nlpost` uses that instead.

```{code-cell} ipython3
def fenrir_nlpost(phi):
    r"Compute the negative loglikihood of :math:`Y_t` using DALTON."
    x0 = phi[n_theta:].reshape((2,1))
    theta = jnp.exp(phi[:n_theta])
    v0 = fitz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    loglik = rf.fenrir(key=key, fun=fitz, W=W, x0=x0, theta=theta, 
                       tmin=tmin, tmax=tmax, n_res=n_res,
                       trans_state=prior_pars['trans_state'], mean_state=prior_pars['mean_state'], var_state=prior_pars['var_state'],
                       trans_obs=trans_obs, mean_obs=mean_obs, var_obs=var_obs, y_obs=y_obs, interrogate=interrogate_tronarp)
    logprior = jnp.sum(jsp.stats.norm.logpdf(
        x=phi,
        loc=phi_mean,
        scale=phi_sd
    ))
    
    return -(loglik + logprior)

def phi_fit(phi_init):
    r"""Compute the optimized :math:`\log{\theta}` and its variance given 
        :math:`Y_t`."""
    n_phi = len(phi_init)
    hes = jacfwd(jacrev(fenrir_nlpost))
    solver = ScipyMinimize(method="Newton-CG", fun = fenrir_nlpost)
    opt_res = solver.run(phi_init)
    phi_hat = opt_res.params
    phi_fisher = hes(phi_hat)
    phi_var = jsp.linalg.solve(phi_fisher, jnp.eye(n_phi))
    return phi_hat, phi_var
```

```{code-cell} ipython3
# optimization process
phi_init = jnp.append(jnp.log(theta_true), ode0)
phi_hat, phi_var = phi_fit(phi_init)
fenrir_post = phi_sample(phi_hat, phi_var, n_samples)
fenrir_post[:, :n_theta] = np.exp(fenrir_post[:, :n_theta])
```

### Dalton

Finally, we present the method, `dalton`, developed by [Wu, Lysy](https://arxiv.org/pdf/2306.05566.pdf). `dalton` uses the model
\begin{equation}\label{eq:dalton}
    \begin{aligned}
        \XX_{n+1} \mid \XX_n & \sim \N(\QQ \XX_n, \RR) \\
        \ZZ_n & \sim \N(\WW \XX_n - \ff(\XX_n, t_n, \tth), \VV_n) \\
        \YY_i & \sim p(\YY_i \mid \XX_{n(i)}, \pph),
    \end{aligned}
\end{equation}
where the data is used directly in the forward pass instead of just the backward pass of `fenrir`. For Gaussian observations such as this example, `dalton.loglikelihood` is the appropriate function to use.

```{code-cell} ipython3
def dalton_nlpost(phi):
    r"Compute the negative loglikihood of :math:`Y_t` using DALTON."
    x0 = phi[n_theta:].reshape((2,1))
    theta = jnp.exp(phi[:n_theta])
    v0 = fitz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    loglik = rd.loglikehood(key=key, fun=fitz, W=W, x0=x0, theta=theta, 
                            tmin=tmin, tmax=tmax, n_res=n_res,
                            trans_state=prior_pars['trans_state'], mean_state=prior_pars['mean_state'], 
                            var_state=prior_pars['var_state'],
                            trans_obs=trans_obs, mean_obs=mean_obs, var_obs=var_obs, 
                            y_obs=y_obs, interrogate=interrogate_tronarp)
    logprior = jnp.sum(jsp.stats.norm.logpdf(
        x=phi,
        loc=phi_mean,
        scale=phi_sd
    ))
    
    return -(loglik + logprior)

def phi_fit(phi_init):
    r"""Compute the optimized :math:`\log{\theta}` and its variance given 
        :math:`Y_t`."""
    n_phi = len(phi_init)
    hes = jacfwd(jacrev(dalton_nlpost))
    solver = ScipyMinimize(method="Newton-CG", fun = dalton_nlpost)
    opt_res = solver.run(phi_init)
    phi_hat = opt_res.params
    phi_fisher = hes(phi_hat)
    phi_var = jsp.linalg.solve(phi_fisher, jnp.eye(n_phi))
    return phi_hat, phi_var
```

```{code-cell} ipython3
# optimization process
phi_init = jnp.append(jnp.log(theta_true), ode0)
phi_hat, phi_var = phi_fit(phi_init)
dalton_post = phi_sample(phi_hat, phi_var, n_samples)
dalton_post[:, :n_theta] = np.exp(dalton_post[:, :n_theta])
```

## Results

We compare the likelihood estimation for the four methods. Only Chrebtii MCMC algorithm differs from the rest because it uses MCMC to sample rather than the Laplacian approximation.

```{code-cell} ipython3
plt.rcParams.update({'font.size': 15})
param_true = np.append(theta_true, ode0)
var_names = ['a', 'b', 'c', r"$V(0)$", r"$R(0)$"]
fig, axs = plt.subplots(1, 5, figsize=(20,5))
for i in range(5):
    sns.kdeplot(basic_post[:, i], ax=axs[i], label='basic')
    sns.kdeplot(mcmc_post[:, i], ax=axs[i], label='mcmc')
    sns.kdeplot(fenrir_post[:, i], ax=axs[i], label='fenrir')
    sns.kdeplot(dalton_post[:, i], ax=axs[i], label='dalton')
    axs[i].axvline(x=param_true[i], linewidth=1, color='r', linestyle='dashed')
    axs[i].set_ylabel("")
    axs[i].set_title(var_names[i])
axs[0].legend(framealpha=0.5, loc='best')
fig.tight_layout()
```

## Dalton Non-Gaussian Observations

Now suppose that the noisy observation model is
\begin{equation}\label{eq:fitznoiseng}
    Y_{ij} \sim \operatorname{Poisson}(\exp{b_0 + b_1x_j(t_i)}),
\end{equation}
where $b_0 = 0.1$ and $b_1 = 0.5$. 

```{code-cell} ipython3
# simulate data
b0 = 0.1
b1 = 0.5
Yt = np.random.default_rng(0).poisson(lam = jnp.exp(b0+b1*exact))
trans_obs = jnp.array([[[1., 0., 0.]], [[1., 0., 0.]]])
y_obs = jnp.expand_dims(Yt, -1)
```

For non-Gaussian observations, please use `dalton.loglikelihood_nn`. The inputs are the same as `dalton.loglikelihood` with `mean_obs` and `var_obs` replaced by `fun_obs`, which is the loglikelihood function of the observation. The rest of the process is the same as the Gaussian example.

```{code-cell} ipython3
def loglik(x_obs, y_curr):
    "Likelihood of observation for DALTON"
    n_block = y_curr.shape[0]
    b0 = 0.1
    b1 = 0.5
    return jnp.sum(jax.vmap(lambda b: jsp.stats.poisson.logpmf(y_curr[b], jnp.exp(b0+b1*x_obs[b])))(jnp.arange(n_block)))

def dalton_nlpost(phi):
    r"Compute the negative loglikihood of :math:`Y_t` using a DALTON."
    x0 = phi[n_theta:].reshape((2,1))
    theta = jnp.exp(phi[:n_theta])
    v0 = fitz(x0, 0, theta)
    x0 = jnp.hstack([x0, v0, jnp.zeros(shape=(x0.shape))])
    lp = rd.loglikehood_nn(key=key, fun=fitz, W=W, x0=x0, theta=theta, 
                            tmin=tmin, tmax=tmax, n_res=n_res,
                            trans_state=prior_pars['trans_state'], mean_state=prior_pars['mean_state'], 
                            var_state=prior_pars['var_state'],
                            fun_obs=loglik, trans_obs=trans_obs, y_obs=y_obs, interrogate=interrogate_tronarp)
    lp += logprior(phi[:n_phi], phi_mean, phi_sd)
    return -lp

def phi_fit(phi_init):
    r"""Compute the optimized :math:`\log{\theta}` and its variance given 
        :math:`Y_t`."""
    n_phi = len(phi_init)
    hes = jacfwd(jacrev(dalton_nlpost))
    solver = ScipyMinimize(method="Newton-CG", fun = dalton_nlpost)
    opt_res = solver.run(phi_init)
    phi_hat = opt_res.params
    phi_fisher = hes(phi_hat)
    phi_var = jsp.linalg.solve(phi_fisher, jnp.eye(n_phi))
    return phi_hat, phi_var
```

```{code-cell} ipython3
# optimization process
phi_init = jnp.append(jnp.log(theta_true), ode0)
phi_hat, phi_var = phi_fit(phi_init)
dalton_post = phi_sample(phi_hat, phi_var, n_samples)
dalton_post[:, :n_theta] = np.exp(dalton_post[:, :n_theta])
```

## Results

```{code-cell} ipython3
fig, axs = plt.subplots(1, 5, figsize=(20,5))
for i in range(5):
    tmp_data = dalton_post[:, i]
    if i==1:
        tmp_data = tmp_data[(tmp_data>0) & (tmp_data<4)]
    sns.kdeplot(tmp_data, ax=axs[i], label='dalton')
    axs[i].axvline(x=param_true[i], linewidth=1, color='r', linestyle='dashed')
    axs[i].set_ylabel("")
    axs[i].set_title(var_names[i])
fig.tight_layout()
```
