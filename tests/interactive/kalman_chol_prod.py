import jax
import jax.numpy as jnp
import jax.scipy as jsp
from rodeo.kalmantv import *

def forward(wgt_meas, mu_meas, var_meas, wgt_state, mu_state, var_state, y):
    r"""
    Forward pass.

    Args:
        wgt_state (ndarray(n_steps-1, n_state, n_state)): Transition matricesin the state model; denoted by :math:`Q_1, \ldots, Q_N`.
        mu_state (ndarray(n_steps, n_state)): Offsets in the state model; denoted by :math:`c_0, \ldots, c_N`.
        var_state (ndarray(n_steps, n_state, n_state)): Variance matrices in the state model; denoted by :math:`R_0, \ldots, R_N`.
        wgt_meas (ndarray(n_steps, n_meas, n_state)): Transition matrices in the measurement model; denoted by :math:`W_0, \ldots, W_N`.
        mu_meas (ndarray(n_steps, n_meas)): Offsets in the measurement model; denoted by :math:`d_0, \ldots, d_N`.
        var_meas (ndarray(n_steps, n_meas, n_meas)): Variance matrices in the measurement model; denoted by :math:`V_0, \ldots, V_N`.
        
    Returns:
        (tuple):
        - **mu_state_pred** (ndarray(n_steps, n_state)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **var_state_pred** (ndarray(n_steps, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        - **mu_state_filt** (ndarray(n_steps, n_state)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        - **var_state_filt** (ndarray(n_steps, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.

    """
    mu_state_init = mu_state[0]
    var_state_init = var_state[0]

    # lax.scan setup
    # scan function
    def scan_fun(carry, for_args):
        mu_state_filt, var_state_filt = carry["state_filt"]
        wgt_state = for_args['wgt_state']
        mu_state = for_args['mu_state']
        var_state = for_args['var_state']
        y = for_args['y']
        mu_meas = for_args['mu_meas']
        wgt_meas = for_args['wgt_meas']
        var_meas = for_args['var_meas']
        
        mu_state_pred, var_state_pred, \
            mu_state_next, var_state_next = filter(
                mu_state_past=mu_state_filt,
                var_state_past=var_state_filt,
                mu_state=mu_state,
                wgt_state=wgt_state,
                var_state=var_state,
                x_meas=y,
                mu_meas=mu_meas,
                wgt_meas=wgt_meas,
                var_meas=var_meas
        )
        # output
        carry = {
            "state_filt": (mu_state_next, var_state_next)
        }
        stack = {
            "state_filt": (mu_state_next, var_state_next),
            "state_pred": (mu_state_pred, var_state_pred)
        }
        return carry, stack

    # scan initial value
    scan_init = {
        "state_filt": (mu_state_init, var_state_init),
    }

    for_args = {
        "wgt_state" : wgt_state[0:],
        "mu_state" : mu_state[1:],
        "var_state" : var_state[1:],
        "y": y[1:],
        "wgt_meas" : wgt_meas[1:],
        "mu_meas" : mu_meas[1:],
        "var_meas" : var_meas[1:]
    }
    # scan itself
    _, scan_out = jax.lax.scan(scan_fun, scan_init, for_args)
    # append initial values to front
    scan_out["state_filt"] = (
        jnp.concatenate([mu_state_init[None], scan_out["state_filt"][0]]),
        jnp.concatenate([var_state_init[None], scan_out["state_filt"][1]])
    )
    scan_out["state_pred"] = (
        jnp.concatenate([mu_state_init[None], scan_out["state_pred"][0]]),
        jnp.concatenate([var_state_init[None], scan_out["state_pred"][1]])
    )
    return scan_out

def backward(mu_state_filt, var_state_filt, 
             mu_state_pred, var_state_pred,
             wgt_state):
    r"""
    Backward pass.

    Args:
        mu_state_pred (ndarray(n_steps, n_state)): Mean estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        var_state_pred (ndarray(n_steps, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t-1] for :math:`t \in [a, b]`.
        mu_state_filt (ndarray(n_steps, n_state)): Mean estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        var_state_filt (ndarray(n_steps, n_state, n_state)): Variance estimate for state at time t given observations from times [a...t] for :math:`t \in [a, b]`.
        wgt_state (ndarray(n_steps-1, n_state, n_state)): Transition matrix defining the solution prior; :math:`Q`.
        
    Returns:
        (tuple):
        - **wgt_state_cond** (ndarray(n_state, n_state)): Transition of smooth conditional at time t given observations from times [0...T]; :math:`A_{n|N}`.
        - **mu_state_cond** (ndarray(n_state)): Offset of smooth conditional at time t given observations from times [0...T]; :math:`b_{n|N}`.
        - *var_state_cond** (ndarray(n_state, n_state)): Variance of smooth conditional at time t given observations from times [0...T]; :math:`V_{n|N}`.

    """
    # Terminal Point
    n_steps = len(mu_state_filt)
    mu_state_end = mu_state_filt[n_steps-1]
    var_state_end = var_state_filt[n_steps-1]

    # backward pass
    # vmap setup
    def vmap_fun(smooth_kwargs):
        mu_state_filt = smooth_kwargs['mu_state_filt']
        var_state_filt = smooth_kwargs['var_state_filt']
        mu_state_pred = smooth_kwargs['mu_state_pred']
        var_state_pred = smooth_kwargs['var_state_pred']
        wgt_state = smooth_kwargs['wgt_state']
        
        wgt_state_cond, mu_state_cond, var_state_cond = smooth_cond(
            mu_state_filt=mu_state_filt,
            var_state_filt=var_state_filt,
            mu_state_pred=mu_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=wgt_state
        )
        return wgt_state_cond, mu_state_cond, var_state_cond

    # scan arguments
    # Slice these arrays so they are aligned.
    # More precisely, for time step t, want filt[t], pred[t+1]
    scan_kwargs = {
        'mu_state_filt': mu_state_filt[:n_steps-1],
        'var_state_filt': var_state_filt[:n_steps-1],
        'mu_state_pred': mu_state_pred[1:n_steps],
        'var_state_pred': var_state_pred[1:n_steps],
        'wgt_state': wgt_state
    }
    wgt_state_cond, mu_state_cond, var_state_cond = jax.vmap(vmap_fun)(scan_kwargs)
    mu_state_cond = jnp.concatenate([mu_state_cond, mu_state_end[None]])
    var_state_cond = jnp.concatenate([var_state_cond, var_state_end[None]])
    return wgt_state_cond, mu_state_cond, var_state_cond

def kalman_chol_prod(J, A, b, C):
    r"""
    Compute :math:`J'\Sigma J` where :math:`\Sigma = var(x_{0:N})`.

    Args:
        J (ndarray(n_steps, n_state)): Random vector to be multiplied.
        A (ndarray(n_steps, n_state, n_state)): Transition of smooth conditional at time t given observations from times [0...T]; :math:`A_{n|N}`.
        b (ndarray(n_steps, n_state)): Offset of smooth conditional at time t given observations from times [0...T]; :math:`b_{n|N}`.
        C (ndarray(n_steps, n_state)): Cholesky of smooth conditional at time t given observations from times [0...T]; :math:`V_{n|N}`.
        
    Returns:
        (float): :math:`J'\SigmaJ`

    """
    # Terminal Point
    n_steps = len(C)
    LJ_last = C[n_steps-1].dot(J[n_steps-1])
    
    # lax.scan setup
    # scan function
    def scan_fun(LJ, args):
        A_n = args['A']
        C_n = args['C']
        J_n = args['J']
        LJ = A_n.dot(LJ) + C_n.dot(J_n)  
        stack = {
            "LJ": LJ
        }
        return LJ, stack

    args = {
        'A': A,
        'C': C[:n_steps-1],
        'J': J[:n_steps-1]
    }
    _, scan_out = jax.lax.scan(scan_fun, LJ_last, args, reverse=True)

    LJ = jnp.concatenate([scan_out['LJ'], LJ_last[None]])
    return LJ
