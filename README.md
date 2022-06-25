# **rodeo:** Probabilistic ODE Solver

*Mohan Wu, Martin Lysy*

---

## Description

**rodeo** is a fast and flexible Python library uses [probabilistic numerics](http://probabilistic-numerics.org/) to solve ordinary differential equations (ODEs).  That is, most ODE solvers (such as [Euler's method](https://en.wikipedia.org/wiki/Euler_method)) produce a deterministic approximation to the ODE on a grid of size `delta`.  As `delta` goes to zero, the approximation converges to the true ODE solution.  Probabilistic solvers such as **rodeo** also output a solution an a grid of size `delta`; however, the solution is random.  Still, as `delta` goes to zero, the probabilistic numerical approximation converges to the true solution. 

**rodeo** implements the probabilistic solver of [Chkrebtii et al (2016)](https://projecteuclid.org/euclid.ba/1473276259). This begins by putting a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) prior on the ODE solution, and updating it sequentially as the solver steps through the grid. **rodeo** is built on `jax` which allows for just-in-time compilation and auto-differentiation. The API of `jax` is almost equivalent to that of `numpy`.

Please note that this is the `jax` only version of **rodeo**. For the legacy versions using various other backends please see [here](https://github.com/mlysy/rodeo).

## Installation

Download the repo from GitHub and then install with the `setup.cfg` script:

```bash
git clone https://github.com/mlysy/rodeo-jax.git
cd rodeo-jax
pip install .
```

## Unit Testing

The unit tests can be ran through the following commands:

```bash
cd tests
python -m unittest discover -v
```

Or, install [**tox**](https://tox.wiki/en/latest/index.html), then from within `rodeo-jax` enter command line: `tox`.

## Usage

Please see the detailed example in the tutorial [introduction](examples/tutorial.ipynb).

## Results

**rodeo** is also capable of performing parameter inference. The main results for three different ODEs found in `/examples/`:

### FitzHugh-Nagumo

![fitzhugh](/docs/figures/fitzfigure.png)

### SEIRAH

![seirah](/docs/figures/seirahfigure.png)

### Hes1

![hes1](/docs/figures/hes1figure.png)
