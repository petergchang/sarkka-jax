import jax.numpy as jnp
import matplotlib.pyplot as plt

# Chapter 3
def plot_linreg(X, Y_tr, Y, estimate=None):
    """Plot linear regression example from Chapter 3.

    Args:
        X: time grid
        Y_tr: "true" signal
        Y: noisy measurements
        estimate (optional): filtered estimates. Defaults to None.
    """    
    plt.figure()
    plt.plot(X, Y, 'k.', label='Measurements')
    plt.plot(X, Y_tr, color='silver', linewidth=7, label="True Signal")
    if estimate is not None:
        plt.plot(X, estimate, color='k', linewidth=3, label="Estimate")
    plt.xlabel('$t$'); plt.ylabel('$y$')
    plt.xlim(0, 1); plt.ylim(0.5, 2)
    plt.yticks(jnp.arange(0.5, 2.1, 0.5))
    plt.gca().set_aspect(0.5)
    plt.legend(loc=1, borderpad=0.8, handlelength=4, fancybox=False, edgecolor='k')
    plt.show()