import numpy as np

def ibm_state(dt, q, sigma):
    """
    Calculate the state transition matrix and variance matrix using the q-times integrated 
    Brownian motion for the Kalman solver.
        
    Args:
        dt (float): The step size between simulation points.
        q: q-times integrated Brownian Motion.
        sigma (float): Parameter in the q-times integrated Brownian Motion.

    Returns:
        (tuple):
        - **A** (ndarray(n_dim_roots, n_dim_roots)): The state transition matrix defined in 
          Kalman solver.
        - **Q** (ndarray(n_dim_roots, n_dim_roots)): The state variance matrix defined in
          Kalman solver.

    """
    A = np.zeros((q+1, q+1), order='F')
    Q = np.zeros((q+1, q+1), order='F')
    for i in range(q+1):
        for j in range(q+1):
            if i<=j:
                A[i, j] = dt**(j-i)/np.math.factorial(j-i)
    
    for i in range(q+1):
        for j in range(q+1):
            num = dt**(2*q+1-i-j)
            denom = (2*q+1-i-j)*np.math.factorial(q-i)*np.math.factorial(q-j)
            Q[i, j] = sigma**2*num/denom
    return A, Q

def ibm_init(dt, n_deriv, sigma):
    """
    Calculates the initial parameters necessary for the Kalman solver with the q-times
    integrated Brownian Motion.

    Args:
        dt (float): The step size between simulation points.
        n_deriv (list(int)): Dimension of the prior.
        sigma (list(float)): Parameter in variance matrix.
        
    Returns:
        (dict):
        - **trans_state** (ndarray(n_block, p, p)) Transition matrix defining the solution prior; :math:`Q_n`.
        - **mean_state** (ndarray(n_block, p)): Transition_offsets defining the solution prior; denoted by :math:`c_n`.
        - **var_state** (ndarray(n_block, p, p)) Variance matrix defining the solution prior; :math:`R_n`.

    """
    n_block = len(n_deriv)
    p = max(n_deriv)
    trans_state = [None]*n_block
    var_state = [None]*n_block
    for i in range(n_block):
        trans_state[i], var_state[i] = ibm_state(dt, n_deriv[i]-1, sigma[i])
    
    #if n_var == 1:
    #    trans_state = trans_state[0]
    #    var_state = var_state[0]
    
    init = {"trans_state":trans_state, "var_state":var_state}
    return init


def indep_init(init, n_deriv):
    """
    Computes the necessary parameters for the Kalman filter and smoother.

    Args:
        init (list(n_var)): Computed initial parameters for each variable.
        n_deriv (list(int)): Number of derivatives for each variable in Kalman solver.
    
    Returns:
        (tuple):
        - **kinit** (dict): Dictionary holding the computed initial parameters for the
          Kalman solver.
    """
    trans_state_i = init['trans_state']
    var_state_i = init['var_state']

    n_var = len(var_state_i)
    p = sum(n_deriv)
    trans_state = np.zeros((p, p), order='F')
    var_state = np.zeros((p, p), order='F')
    ind = 0
    for i in range(n_var):
        trans_state[ind:ind+n_deriv[i], ind:ind+n_deriv[i]] = trans_state_i[i]
        var_state[ind:ind+n_deriv[i], ind:ind+n_deriv[i]] = var_state_i[i]
        ind += n_deriv[i]
    kinit = {"trans_state":trans_state, "var_state":var_state}
    
    return kinit
