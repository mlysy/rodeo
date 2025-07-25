# Change log

## rodeo 1.1.3

* Breaking changes:
    * `prior_weight`, `prior_var` arguments are combined as `prior_pars` for all solver and inference methods
    * `rodeo.inference.pseudo_marginal` replaces `rodeo.inference.random_walk_aux`. The API is now exactly like BlackJAX, except that the `init` method requires a PRNG-key. The signature for the log-posterior/likelihood must be `parameters`, `key`. 

* Minor change:
    * Blackjax version is no longer fixed for Python 3.10+.

## rodeo 1.1.2

* Setup is done only using `pyproject.toml` and `setup.cfg` is no longer needed.

## rodeo 1.1.1

* No changes besides minor documentation edits.

## rodeo 1.1.0

* Breaking changes:
    * `kalman_funs` argument is changed to `kalman_type`. rodeo supports standard Kalman and square-root Kalman and these are directly implemented in the library. The `kalman_type` argument picks the algorithm to use.

* New features:
    * Added `first_order_pad` to help users with zero-padding the initial value and `ode_weight` matrix.

## rodeo 1.0.0

* Breaking changes:
    * rodeo now depends on jax and previous Cython/C++ implementations are completely removed.
    * Standard Kalman algorithm is now under `rodeo.kalmantv.standard` instead of just `rodeo.kalmantv`.
    * The ODE solver in rodeo is now under `rodeo.solve`.
    * The IBM prior only returns `wgt_state`, and `var_state`. That is, `mu_state` is assumed to be 0.

* New features:
    * Added new parameter inference algorithms in `rodeo.inference`: `basic`, `dalton`, `fenrir`, `random_walk_aux`, `magi`.
    * Added square-root Kalman algorithms: `rodeo.kalmantv.square_root`
    * Added new interrogation functions from new research.
    * All ODE solver and parameter inference methods support block-wise computation for efficiency. This is the default option if the IBM prior is used from `rodeo.prior.ibm`.

## rodeo 0.4

* Initial release of rodeo: a probabilistic ODE solver based on the Bayesian filtering paradigm with Python frontend and three backends:
    * C++ using Eigen
    * Cython using BLAS/LAPACK
    * numba using BLAS/LAPACK
* Two methods for parameterizing the Gaussian Markov prior are included: IBM and CAR.
* Three interrogation functions are included: Chkrebtii, Schobert, and a mix of the two called rodeo.
* Kalman filtering/smoothing algorithms are written in C++ using Eigen.
* This version is still available at [rodeo-legacy](https://github.com/mlysy/rodeo-legacy).
