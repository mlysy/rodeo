r"This module is for plotting the figures in the paper"
# math
from math import ceil
import numpy as np
import scipy as sp

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def theta_plot(theta_euler, theta_kalman, theta_diffrax, theta_true, step_sizes, var_names, clip=None, rows=1):
    r"""Plot the distribution of :math:`\theta` using the Kalman solver 
        and the Euler approximation."""
    n_hlst, _, n_theta = theta_euler.shape
    ncol = ceil(n_theta/rows) +1
    nrow = 2
    fig = plt.figure(figsize=(20, 10*rows))
    patches = [None]*(n_hlst+2)
    if clip is None:
        clip = [None]*ncol*rows 
    carry = 0
    for t in range(1,n_theta+1):
        row = (t-1)//(ncol-1)
        if t%(ncol)==0:
            carry +=1
        
        axs1 = fig.add_subplot(rows*nrow, ncol, t+row*(ncol)+carry)
        axs2 = fig.add_subplot(rows*nrow, ncol, t+(row+1)*(ncol)+carry)
        axs2.get_shared_x_axes().join(axs2, axs1)
        axs1.set_title(var_names[t-1])
        if (t+carry)%ncol==1:
            axs1.set_ylabel('Euler')
            axs2.set_ylabel('rodeo')
        
        for axs in [axs1, axs2]:
            axs.axvline(x=theta_true[t-1], linewidth=1, color='r', linestyle='dashed')
            axs.set_yticks([])

        for h in range(n_hlst):
            if t==1:
                patches[h] = mpatches.Patch(color='C{}'.format(h), label='$\\Delta$ t ={}'.format(step_sizes[h]))
            sns.kdeplot(theta_euler[h, :, t-1], ax=axs1, clip=clip[t-1])
            sns.kdeplot(theta_kalman[h, :, t-1], ax=axs2, clip=clip[t-1])
        

        sns.kdeplot(theta_diffrax[:, t-1], ax=axs1, color='black', clip=clip[t-1])
        sns.kdeplot(theta_diffrax[:, t-1], ax=axs2, color='black', clip=clip[t-1])
        if t==n_theta:
            patches[-2] = mpatches.Patch(color='black', label="True Posterior")
            patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\theta$')
            
    fig.legend(handles=patches, framealpha=0.5, loc=7)
    
    fig.tight_layout()
    plt.show()
    return fig

def theta_plotsingle(theta, theta_diffrax, theta_true, step_sizes, var_names, clip=None, rows=1):
    r"""Plot the distribution of :math:`\theta` using the Kalman solver 
        and the Euler approximation."""
    n_hlst, _, n_theta = theta.shape
    ncol = ceil(n_theta/rows) +1
    fig = plt.figure(figsize=(20, 5*rows))
    patches = [None]*(n_hlst+2)
    if clip is None:
        clip = [None]*ncol*rows 
    carry = 0
    for t in range(1,n_theta+1):
        row = (t-1)//(ncol-1)
        if t%(ncol)==0:
            carry +=1
        
        axs = fig.add_subplot(rows, ncol, t+carry)
        axs.set_title(var_names[t-1])
        axs.axvline(x=theta_true[t-1], linewidth=1, color='r', linestyle='dashed')
        axs.set_yticks([])

        for h in range(n_hlst):
            if t==1:
                patches[h] = mpatches.Patch(color='C{}'.format(h), label='$\\Delta$ t ={}'.format(step_sizes[h]))
            sns.kdeplot(theta[h, :, t-1], ax=axs, clip=clip[t-1])
        
        sns.kdeplot(theta_diffrax[:, t-1], ax=axs, color='black', clip=clip[t-1])
        
        if t==n_theta:
            patches[-2] = mpatches.Patch(color='black', label="True Posterior")
            patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\Theta$')
            
    fig.legend(handles=patches, framealpha=0.5, loc=7)
    
    fig.tight_layout()
    plt.show()
    return fig
