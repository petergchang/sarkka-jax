import jax.numpy as jnp
import matplotlib.pyplot as plt

# Chapter 3
def plot_linreg(X, Y_tr, Y, estimate=None):    
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

def plot_linreg_mean_convergence(X, rmean1, rmean2, bmean1, bmean2):
    plt.figure()
    plt.plot(X, rmean1, 'k', linewidth=3, label=r"Recursive $E[\theta_1]$")
    plt.plot(X, bmean1, 'k--', linewidth=3, label=r"Batch $E[\theta_1]$")
    plt.plot(X, rmean2, color='gray', linewidth=3, label=r"Recursive $E[\theta_2]$")
    plt.plot(X, bmean2, '--', color='gray', linewidth=3, label=r"Batch $E[\theta_2]$")
    plt.xlabel('$t$'); plt.ylabel('$y$')
    plt.xlim(0, 1); plt.ylim(-0.4, 1.2)
    plt.yticks(jnp.arange(-0.4, 1.3, 0.2))
    plt.gca().set_aspect(0.5)
    plt.legend(loc=4, borderpad=0.8, handlelength=4, fancybox=False, edgecolor='k');
    plt.show()

def plot_linreg_var_convergence(X, rvar1, rvar2, bvar1, bvar2):
    plt.figure()
    plt.plot(X, rvar1, 'k', linewidth=3, label=r"Recursive $Var[\theta_1]$")
    plt.plot(X, bvar1, 'k--', linewidth=3, label=r"Batch $Var[\theta_1]$")
    plt.plot(X, rvar2, color='gray', linewidth=3, label=r"Recursive $Var[\theta_2]$")
    plt.plot(X, bvar2, '--', color='gray', linewidth=3, label=r"Batch $Var[\theta_2]$")
    plt.xlabel('$t$'); plt.ylabel('$y$')
    plt.yscale("log")
    plt.xlim(0, 1); plt.ylim(1e-3, 5)
    plt.gca().set_aspect(0.2)
    plt.legend(loc=1, borderpad=0.8, handlelength=4, fancybox=False, edgecolor='k');
    plt.grid(True, which="both", ls=':')

def plot_sine(X, Y_tr, Y, X_est=None, Y_est=None, xrlim=2., loc=1):
    if X_est is None:
        X_est = X
    plt.figure()
    plt.plot(X, Y, 'k.', label='Measurements')
    plt.plot(X, Y_tr, color='silver', linewidth=7, label="True Signal")
    if Y_est is not None:
        plt.plot(X_est, Y_est, 'k-', linewidth=2, label="Estimate")
    plt.xlabel('$t$'); plt.ylabel('$y$')
    plt.xlim(0, 2); plt.ylim(-1.5, 1.5)
    plt.xticks(jnp.arange(0, xrlim+0.1, 0.5))
    plt.yticks(jnp.arange(-1.5, 1.6, 0.5))
    plt.gca().set_aspect(0.5)
    plt.legend(loc=loc, borderpad=0.8, handlelength=4, fancybox=False, edgecolor='k');
    plt.show()