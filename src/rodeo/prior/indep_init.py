r"""
This module combines prior parameters into a single block.
"""

import jax
import jax.scipy.linalg as jsl

def indep_init(prior_pars):
    """
    Combine blocks of prior parameters into dense matrices.

    Args:
        prior_pars (tuple): A tuple containing the weight matrix and the variance matrix defining the solution prior; :math:`Q, R`.

    Returns:
        (tuple):
        - **prior_weight** (ndarray(1, n_block * p, n_block * p)): Transition matrix defining the solution prior; :math:`Q`.
        - **prior_var** (ndarray(1, n_block * p, n_block * p)): Variance matrix defining the solution prior; :math:`R`.
    """
    prior_weight, prior_var = prior_pars
    prior_weight = jsl.block_diag(*prior_weight)[None, :]
    prior_var = jsl.block_diag(*prior_var)[None, :]
    return prior_weight, prior_var
