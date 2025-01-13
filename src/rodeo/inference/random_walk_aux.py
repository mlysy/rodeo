"""
Implements the user interfaces for Random Walk Rosenbluth-Metropolis-Hastings kernels with auxiliary variables.
That is, the `logdensity_fn` returns a tuple of which the first element is the log-density, and the second
is a PyTree containing the auxiliary variables.
The remainder of this file is nearly identical to `random_walk.py`. 

Some interfaces are exposed here for convenience and for entry level users, who might be familiar
with simpler versions of the algorithms, but in all cases they are particular instantiations
of the Random Walk Rosenbluth-Metropolis-Hastings.

Let's note $x_{t-1}$ to the previous position and $x_t$ to the newly sampled one.

The variants offered are:

1. Proposal distribution as addition of random noice from previous position. This means
   $x_t = x_{t-1} + step$.

    Function: `additive_step`

2. Independent proposal distribution: $P(x_t)$ doesn't depend on $x_{t_1}$.

    Function: `irmh`

3. Proposal distribution using a symmetric function. That means $P(x_t|x_{t-1}) = P(x_{t-1}|x_t)$.
   See 'Metropolis Algorithm' in [1].

    Function: `rmh` without proposal_logdensity_fn.

4. Asymmetric proposal distribution. See 'Metropolis-Hastings' Algorithm in [1].

    Function: `rmh` with proposal_logdensity_fn.

    
Reference: Andrew Gelman, John B Carlin, Hal S Stern, and Donald B Rubin. Bayesian data analysis. Chapman and Hall/CRC, 2014. Section 11.2

Example:
    
    The simplest case is:

    .. code::

        random_walk = blackjax.additive_step_random_walk(logdensity_fn, blackjax.mcmc.random_walk.normal(sigma))
        state = random_walk.init(position)
        new_state, info = random_walk.step(rng_key, state)

    In all cases we can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(random_walk.step)
        new_state, info = step(rng_key, state)

"""
from typing import Callable, NamedTuple, Optional

import jax
from jax import numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc import proposal
from blackjax.types import Array, ArrayLikeTree, ArrayTree, PRNGKey
from blackjax.util import generate_gaussian_noise
from blackjax.mcmc.random_walk import normal as normal

# ignore this for now
# __all__ = [
#     "build_additive_step",
#     "normal",
#     "build_irmh",
#     "build_rmh",
#     "RWInfo",
#     "RWState",
#     "rmh_proposal",
#     "build_rmh_transition_energy",
#     "additive_step_random_walk",
#     "irmh_as_top_level_api",
#     "rmh_as_top_level_api",
#     "normal_random_walk",
# ]


# imported from blackjax.mcmc.random_walk
# def normal(sigma: Array) -> Callable:
#     """Normal Random Walk proposal.

#     Propose a new position such that its distance to the current position is
#     normally distributed. Suitable for continuous variables.

#     Parameter
#     ---------
#     sigma:
#         vector or matrix that contains the standard deviation of the centered
#         normal distribution from which we draw the move proposals.

#     """
#     if jnp.ndim(sigma) > 2:
#         raise ValueError("sigma must be a vector or a matrix.")

#     def propose(rng_key: PRNGKey, position: ArrayLikeTree) -> ArrayTree:
#         return generate_gaussian_noise(rng_key, position, sigma=sigma)

#     return propose


class RWAState(NamedTuple):
    """
    State of the Random Walk Auxiliary (RWA) chain.

    Attributes:
        position (ArrayTree): Current position of the chain.
        logdensity (float): Current value of the log-density.
        auxdata (ArrayTree, optional): Current value of the auxiliary data.
    """

    position: ArrayTree
    logdensity: float
    auxdata: ArrayTree = None


class RWAInfo(NamedTuple):
    """
    Additional information about the RWA chain step.

    Attributes:
        acceptance_rate (float): The acceptance probability of the proposed transition.
        is_accepted (bool): Indicates whether the proposed state was accepted.
        proposal (RWAState): The proposed state of the chain.
    """

    acceptance_rate: float
    is_accepted: bool
    proposal: RWAState


def init(position: ArrayLikeTree, logdensity_fn: Callable) -> RWAState:
    """
    Create an initial chain state from a given position.

    Args:
        position (ArrayLikeTree): The initial position of the chain.
        logdensity_fn (Callable): Function to compute the log-probability density of the distribution.

    Returns:
        (RWAState): The initialized state of the chain.
    """
    logdensity, auxdata = logdensity_fn(position)
    return RWAState(position, logdensity, auxdata)


def build_additive_step():
    """"
    Build a Random Walk Rosenbluth-Metropolis-Hastings (RMH) kernel using an additive step proposal.

    Returns:
        (Callable): A function that takes a random key and chain state, performs an RMH step, and returns the new state and transition info.
    """

    def kernel(
        rng_key: PRNGKey, state: RWAState, logdensity_fn: Callable, random_step: Callable
    ) -> tuple[RWAState, RWAInfo]:
        def proposal_generator(key_proposal, position):
            move_proposal = random_step(key_proposal, position)
            new_position = jax.tree_util.tree_map(
                jnp.add, position, move_proposal)
            return new_position

        inner_kernel = build_rmh()
        return inner_kernel(rng_key, state, logdensity_fn, proposal_generator)

    return kernel


def normal_random_walk(logdensity_fn: Callable, sigma):
    """
    Create a Gaussian additive step random walk Metropolis-Hastings sampler.

    This method initializes a random walk sampler with Gaussian-distributed steps.

    Args:
        logdensity_fn (Callable): Function to compute the log-probability density of the distribution.
        sigma (ArrayLikeTree): Standard deviation of the Gaussian distribution used for the proposal steps.

    Returns:
        (SamplingAlgorithm): An object with `init` and `step` methods to run the Gaussian RMH sampler.

    """
    return additive_step_random_walk(logdensity_fn, normal(sigma))


def additive_step_random_walk(
    logdensity_fn: Callable, random_step: Callable
) -> SamplingAlgorithm:
    """Implements the user interface for the Additive Step RMH

    Example:

    A new kernel can be initialized and used with the following code:

    .. code::

        rw = blackjax.additive_step_random_walk(logdensity_fn, random_step)
        state = rw.init(position)
        new_state, info = rw.step(rng_key, state)

    The specific case of a Gaussian `random_step` is already implemented, either with independent components
    when `covariance_matrix` is a one dimensional array or with dependent components if a two dimensional array:

    .. code::

        rw_gaussian = blackjax.additive_step_random_walk.normal_random_walk(logdensity_fn, covariance_matrix)
        state = rw_gaussian.init(position)
        new_state, info = rw_gaussian.step(rng_key, state)

    Args:
        logdensity_fn (Callable): Function to compute the log-probability density of the distribution.
        random_step (Callable): A function that generates a step to be added to the current state. This
            function takes a PRNG key and the current position as input and returns a new proposal step.

    Returns:
        (SamplingAlgorithm): A sampling algorithm with `init` and `step` methods to perform RMH sampling.
    """
    kernel = build_additive_step()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(rng_key, state, logdensity_fn, random_step)

    return SamplingAlgorithm(init_fn, step_fn)


def build_irmh() -> Callable:
    """
    Build an Independent Random Walk Rosenbluth-Metropolis-Hastings (RMH) kernel.

    This kernel uses a proposal distribution that is independent of the current state, i.e., 
    the new proposed state is sampled independently of the particle's current position.

    Returns:
        (Callable): A function (kernel) that takes a PRNG key and a PyTree containing the 
        current state of the chain and that returns a new state of the chain along with
        information about the transition.
    """

    def kernel(
        rng_key: PRNGKey,
        state: RWAState,
        logdensity_fn: Callable,
        proposal_distribution: Callable,
        proposal_logdensity_fn: Optional[Callable] = None,
    ) -> tuple[RWAState, RWAInfo]:
        """
        Args:
            proposal_distribution (Callable): A function that takes a PRNG key and returns a 
                sample in the same domain as the target distribution.
            proposal_logdensity_fn (Optional[Callable]): A function that returns the log-density
                of obtaining a given proposal, given the current state. This is required
                for non-symmetric proposals where P(x_t | x_{t-1}) ≠ P(x_{t-1} | x_t). If not
                provided, the proposal is assumed to be symmetric.
        """

        def proposal_generator(rng_key: PRNGKey, position: ArrayTree):
            del position
            return proposal_distribution(rng_key)

        inner_kernel = build_rmh()
        return inner_kernel(
            rng_key, state, logdensity_fn, proposal_generator, proposal_logdensity_fn
        )

    return kernel


def irmh_as_top_level_api(
    logdensity_fn: Callable,
    proposal_distribution: Callable,
    proposal_logdensity_fn: Optional[Callable] = None,
) -> SamplingAlgorithm:
    """Implements the (basic) user interface for the independent RMH.

    Example:

    A new kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.irmh(logdensity_fn, proposal_distribution)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    Args:
    logdensity_fn (Callable): The log-probability density function of the distribution to sample from.
    proposal_distribution (Callable): A function that takes a PRNG key and produces a new proposal.
        The proposal is independent of the current state of the sampler.
    proposal_logdensity_fn (Optional[Callable]): A function that returns the log-density of obtaining
        a given proposal, given the current state. This is required for non-symmetric proposals.
        If not provided, the proposal is assumed to be symmetric.

    Returns:
        (SamplingAlgorithm): An object containing `init` and `step` methods for performing
        Independent Random Walk Metropolis-Hastings sampling.

    """
    kernel = build_irmh()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            proposal_distribution,
            proposal_logdensity_fn,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def build_rmh():
    """Build a Rosenbluth-Metropolis-Hastings kernel.

    Returns:
        (Callable): A function (kernel) that takes a PRNG key and a PyTree containing the 
        current state of the chain and that returns a new state of the chain along with
        information about the transition.

    """

    def kernel(
        rng_key: PRNGKey,
        state: RWAState,
        logdensity_fn: Callable,
        transition_generator: Callable,
        proposal_logdensity_fn: Optional[Callable] = None,
    ) -> tuple[RWAState, RWAInfo]:
        """
        Perform one step of the Rosenbluth-Metropolis-Hastings (RMH) algorithm.

        This function moves the Markov chain by one step using the RMH algorithm by generating a candidate state and accepting or rejecting it based on the Metropolis-Hastings acceptance criterion.

        Args:
            rng_key (PRNGKey): The pseudo-random number generator key used to generate random numbers.
            state (RWAState): The current state of the Markov chain.
            logdensity_fn (Callable): A function that computes the log-probability at a given position.
            transition_generator (Callable): A function that generates a candidate transition for the Markov chain.
            proposal_logdensity_fn (Optional[Callable]): A function that returns the log-density of obtaining a given proposal from the current state. Required for non-symmetric proposals. If not provided, the proposal is assumed to be symmetric.

        Returns:
            (tuple):
                - **RWAState**: The new state of the Markov chain after the RMH step.
                - **RWAInfo**: Additional information about the step, such as the acceptance probability and whether the proposal was accepted.

        """
        transition_energy = build_rmh_transition_energy(proposal_logdensity_fn)

        compute_acceptance_ratio = proposal.compute_asymmetric_acceptance_ratio(
            transition_energy
        )

        proposal_generator = rmh_proposal(
            logdensity_fn, transition_generator, compute_acceptance_ratio
        )
        new_state, do_accept, p_accept = proposal_generator(rng_key, state)
        return new_state, RWAInfo(p_accept, do_accept, new_state)

    return kernel


def rmh_as_top_level_api(
    logdensity_fn: Callable,
    proposal_generator: Callable[[PRNGKey, ArrayLikeTree], ArrayTree],
    proposal_logdensity_fn: Optional[Callable[[
        ArrayLikeTree], ArrayTree]] = None,
) -> SamplingAlgorithm:
    """Implements the user interface for the RMH.

    Example:

    A new kernel can be initialized and used with the following code:

    .. code::

        rmh = blackjax.rmh(logdensity_fn, proposal_generator)
        state = rmh.init(position)
        new_state, info = rmh.step(rng_key, state)

    We can JIT-compile the step function for better performance

    .. code::

        step = jax.jit(rmh.step)
        new_state, info = step(rng_key, state)

    Create a user interface for the Rosenbluth-Metropolis-Hastings (RMH) sampler.

    This function returns a `SamplingAlgorithm` object that provides `init` and `step` methods for performing RMH sampling. The user can specify a custom proposal generator and an optional log-density function for non-symmetric proposals.

    Args:
        logdensity_fn (Callable): The log-probability density function of the distribution to sample from.
        proposal_generator (Callable): A function that takes a random number generator key and the current state, then generates a new proposal.
        proposal_logdensity_fn (Optional[Callable]): The log-density function associated with the proposal generator. If the proposal distribution is non-symmetric (i.e., P(x_t | x_{t-1}) ≠ P(x_{t-1} | x_t)), this must be provided to apply the Metropolis-Hastings correction for detailed balance.

    Returns:
        (SamplingAlgorithm): An object containing `init` and `step` methods for running the RMH sampler.

    """
    kernel = build_rmh()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position, logdensity_fn)

    def step_fn(rng_key: PRNGKey, state):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            proposal_generator,
            proposal_logdensity_fn,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def build_rmh_transition_energy(proposal_logdensity_fn: Optional[Callable]) -> Callable:
    if proposal_logdensity_fn is None:

        def transition_energy(prev_state, new_state):
            return -new_state.logdensity

    else:

        def transition_energy(prev_state, new_state):
            return -new_state.logdensity - proposal_logdensity_fn(new_state, prev_state)

    return transition_energy


def rmh_proposal(
    logdensity_fn: Callable,
    transition_distribution: Callable,
    compute_acceptance_ratio: Callable,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    """
    Args:
        logdensity_fn (Callable): The log-probability density function of the distribution to sample from.
        transition_distribution (Callable): A function that takes a random number generator key and the current state, then generates a new proposal.
        compute_acceptance_ratio (Callable): A function to compute the acceptance ratio.
        sample_proposal (Callable): A function to generate the next sample given proposal and previous state.
    
    Returns:
        (Callable): Generator for sample proposals.
    """

    def generate(rng_key, previous_state: RWAState) -> tuple[RWAState, bool, float]:
        key_proposal, key_accept = jax.random.split(rng_key, 2)
        position, _, _ = previous_state
        new_position = transition_distribution(key_proposal, position)
        new_logdensity, new_auxdata = logdensity_fn(new_position)
        proposed_state = RWAState(new_position, new_logdensity, new_auxdata)
        log_p_accept = compute_acceptance_ratio(previous_state, proposed_state)
        accepted_state, info = sample_proposal(
            key_accept, log_p_accept, previous_state, proposed_state
        )
        do_accept, p_accept, _ = info
        return accepted_state, do_accept, p_accept

    return generate
