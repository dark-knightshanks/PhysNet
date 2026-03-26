import core
import jax
import jax.numpy as jnp
from flax import nnx
jnp.random.seed(69)
rngs = nnx.Rngs(0)
in_dim = 0
hidden_dims = 0
q = [], q_dot = []

mass_net = core.MNN(in_dim, hidden_dims, rngs=rngs)
damp_net = core.DNN(in_dim, hidden_dims, rngs=rngs)
control_net = mass_net = core.ANN(in_dim, hidden_dims, rngs=rngs)
potential_net = mass_net = core.PNN(in_dim, hidden_dims, rngs=rngs)

M = mass_net(q)
D = damp_net(q)
A = control_net(q)
V = potential_net(q)

def lagrange():
    T = 0.5 * q_dot.T * M
    L = T - V
    return L

def lagrangian_eq(lagrangian, state, u, params):
    q, q_dot = jnp.split(state, 2)
    # using Euler-Lagrange equaition
    
    mi_matrix = M  # mass inertia matrix
    # solving accleration using euler lagrange
    q_ddot = (jnp.linalg.pinv(mi_matrix)) @ (A @ u + 
        jnp.jacobian(lagrangian, 0)(q, q_dot) - 
        (jnp.jacobian(jnp.jacobian(lagrangian, 1), 0)(q, q_dot)) @ q_dot - D @ q_dot
    )
    return q_ddot



