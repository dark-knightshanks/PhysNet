import core
import jax
import jax.numpy as jnp
from flax import nnx

rngs       = nnx.Rngs(0)
in_dim     = 2          # set to your actual DOF (e.g. 2 for double pendulum)
hidden_dims = [64, 64]
M_ctrl     = 1          # control input dimension
q = jnp.zeros(in_dim)
q_dot = jnp.zeros(in_dim)

mass_net = core.MNN(in_dim, hidden_dims, rngs=rngs)
damp_net = core.DNN(in_dim, hidden_dims, rngs=rngs)
control_net =  core.ANN(in_dim,M_ctrl, hidden_dims, rngs=rngs)
potential_net  = core.PNN(in_dim, hidden_dims, rngs=rngs)


def lagrangian(q, q_dot):
    M = mass_net(q)                        # [N, N]  pos-def inertia
    V = potential_net(q)                   # scalar
    T = 0.5 * q_dot @ M @ q_dot           # scalar  (was: q_dot.T * M — wrong)
    return T - V

def lagrangian_eq(q, q_dot, u):

    # using Euler-Lagrange equaition
      # Recompute structure matrices at current q
    M = mass_net(q)        # [N, N]
    D = damp_net(q)        # [N, N]
    A = control_net(q)     # [N, M_ctrl]

    # dL/dq  — gradient of lagrangian w.r.t. position
    dLdq = jax.grad(lagrangian, argnums=0)(q, q_dot)          # [N]

    # d/dt(dL/dq_dot) = d/dq(dL/dq_dot) @ q_dot
    # jax.jacobian(f, argnums=1) gives d(dL/dq_dot)/dq — shape [N, N]
    dLdqdot     = jax.grad(lagrangian, argnums=1)              # q_dot gradient fn
    d_dLdqdot_dq = jax.jacobian(dLdqdot, argnums=0)(q, q_dot) # [N, N]
    coriolis    = d_dLdqdot_dq @ q_dot                        # [N]

    # Euler-Lagrange: M q'' = Au - Dq' + dL/dq - d/dt(dL/dq')
    rhs = A @ u + dLdq - coriolis - D @ q_dot                 # [N]

    # Solve for acceleration — valid since M is pos-def by construction
    q_ddot = jnp.linalg.solve(M, rhs)                         # [N]

    return q_ddot

# ── Quick smoke test ──
u = jnp.zeros(M_ctrl)
q_ddot = lagrangian_eq(q, q_dot, u)
print("q_ddot shape:", q_ddot.shape)   # should be (2,)
