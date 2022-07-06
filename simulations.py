import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jax import lax

# 1-dimensional random walk simulation
def simulate_rw_1d(init, Q, R, num_steps, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    def _step(carry, rng):
        x_prev = carry
        key1, key2 = jr.split(rng)

        # Random walk and measurement
        x_post = x_prev + jr.normal(key1)*Q
        y = x_post + jr.normal(key2)
        return x_post, (x_post, y)

    carry = init
    rngs = jr.split(key, num_steps)
    _, (xs, ys) = lax.scan(
        _step, carry, rngs
    )
    return xs, ys

def simulate_rw_1d_with_default_params():
    return simulate_rw_1d(0, 1, 1, num_steps=100)

# Car trajectory simulation (Example 3.6)
def simulate_trajectory(m_0, A, Q, H, R, num_steps, key=42):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    M, N = m_0.shape[-1], R.shape[-1]

    def _step(carry, rng):
        state = carry
        rng1, rng2 = jr.split(rng, 2)
        
        next_state = A @ state + jr.multivariate_normal(rng1, jnp.zeros(M), Q)
        observation = H @ state + jr.multivariate_normal(rng2, jnp.zeros(N), R)
        return next_state, (state, observation)

    rngs = jr.split(key, num_steps)
    _, (states, observations) = lax.scan(
        _step, m_0, rngs
    )
    return states, observations

def simulate_trajectory_with_default_params():
    m_0 = jnp.array([0., 0., 1., -1.])
    dt = 0.1
    q1, q2 = 1, 1
    rsig1, rsig2 = 0.5, 0.5
    A = jnp.array([[1, 0, dt,  0],
                   [0, 1,  0, dt],
                   [0, 0,  1,  0],
                   [0, 0,  0,  1]])
    Q = jnp.array([[q1*dt**3/3,          0, q1*dt**2/2,          0],
                   [         0, q2*dt**3/3,          0, q2*dt**2/2],
                   [q1*dt**2/2,          0,      q1*dt,          0],
                   [         0, q2*dt**2/2,          0,      q2*dt]])
    H = jnp.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]])
    R = jnp.array([[rsig1**2,        0],
                   [       0, rsig2**2]])
    return simulate_trajectory(m_0, A, Q, H, R, num_steps=100)

# Pendulum simulation (Example 3.7)
def simulate_pendulum(m_0, f, h, Q, R, num_steps, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    M = m_0.shape[0]

    def _step(carry, rng):
        state = carry
        rng1, rng2 = jr.split(rng, 2)

        next_state = f(state) + jr.multivariate_normal(rng1, jnp.zeros(M), Q)
        obs = h(next_state) + jr.normal(rng2) * R
        return next_state, (next_state, obs)

    rngs = jr.split(key, num_steps)
    _, (states, observations) = lax.scan(
        _step, m_0, rngs
    )
    return states, observations

def pendulum_default_params(dt=0.0125):
    m_0 = jnp.array([jnp.pi/2, 0])
    P_0 = jnp.eye(2) * 0.1
    dt = dt
    q = 1
    g = 9.8
    Q = jnp.array([[q*dt**3/3, q*dt**2/2],
                [q*dt**2/2,      q*dt]])
    R = 0.3
    f = lambda x: jnp.array([x[0] + x[1]*dt, x[1] - g*jnp.sin(x[0])*dt])
    h = lambda x: jnp.array([jnp.sin(x[0])])
    return (m_0, P_0, f, h, Q, R)

def simulate_pendulum_with_default_params(dt=0.0125):
    m_0, _, f, h, Q, R = pendulum_default_params(dt=dt)
    return simulate_pendulum(m_0, f, h, Q, R, num_steps=400)