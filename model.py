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


class LNN(nnx.Module):
    def __init__(self,N,M_ctrl, hidden_dims, rngs):
        self.mass_net = core.MNN(N, hidden_dims, rngs=rngs)
        self.damp_net = core.DNN(N, hidden_dims, rngs=rngs)
        self.control_net =  core.ANN(N, M_ctrl, hidden_dims, rngs=rngs)
        self.potential_net  = core.PNN(N, hidden_dims, rngs=rngs)
        self.N = N

    def lagrangian(self, q, q_dot):
        M = self.mass_net(q)                        # [N, N]  pos-def inertia
        V = self.potential_net(q)                   # scalar
        T = 0.5 * q_dot @ M @ q_dot           # scalar  (was: q_dot.T * M — wrong)
        return T - V

    def __call__(self, q, q_dot, u):

    # using Euler-Lagrange equaition
      # Recompute structure matrices at current q
        M = self.mass_net(q)        # [N, N]
        D = self.damp_net(q)        # [N, N]
        A = self.control_net(q)     # [N, M_ctrl]

        # dL/dq  — gradient of lagrangian w.r.t. position
        dLdq = jax.grad(self.lagrangian, argnums=0)(q, q_dot)          # [N]

        # d/dt(dL/dq_dot) = d/dq(dL/dq_dot) @ q_dot
        # jax.jacobian(f, argnums=1) gives d(dL/dq_dot)/dq — shape [N, N]
        dLdqdot     = jax.grad(self.lagrangian, argnums=1)              # q_dot gradient fn
        d_dLdqdot_dq = jax.jacobian(dLdqdot, argnums=0)(q, q_dot) # [N, N]
        coriolis    = d_dLdqdot_dq @ q_dot                        # [N]

        # Euler-Lagrange: M q'' = Au - Dq' + dL/dq - d/dt(dL/dq')
        rhs = A @ u + dLdq - coriolis - D @ q_dot                 # [N]

        # Solve for acceleration — valid since M is pos-def by construction
        q_ddot = jnp.linalg.solve(M, rhs)                         # [N]

        return q_ddot

# ── Quick smoke test ──
lnn_model = LNN(in_dim, M_ctrl, hidden_dims, rngs)   # create the object
u         = jnp.zeros(M_ctrl)
q_ddot    = lnn_model(q, q_dot, u)                   # now call it
print("q_ddot shape:", q_ddot.shape)   # should be (2,)

class HNN(nnx.Module):
    def __init__(self, N, M_ctrl, hidden_dims, rngs):
        self.mass_net = core.MNN(N, hidden_dims, rngs=rngs)
        self.damp_net = core.DNN(N, hidden_dims, rngs=rngs)
        self.control_net =  core.ANN(N, M_ctrl, hidden_dims, rngs=rngs)
        self.potential_net  = core.PNN(N, hidden_dims, rngs=rngs)
        self.N = N

    def hamiltonina(self, q, p):
        M = self.mass_net(q)                        # [N, N]  pos-def inertia
        V = self.potential_net(q)                   # scalar
        Minv_p = jnp.linalg.solve(M, p)
        T   = 0.5 * p @ Minv_p          # scalar  (was: q_dot.T * M — wrong)
        return T + V
    
    def __call__(self, q, q_dot, u):
        M = self.mass_net(q)        # [N, N]
        D = self.damp_net(q)        # [N, N]
        A = self.control_net(q)     # [N, M_ctrl]
        # Convert velocity to momentum: p = M(q) @ q_dot
        p = M @ q_dot        
        # Hamilton's equations via autodiff
        # dH/dp gives q_dot (velocity from momentum)
        # dH/dq gives the conservative force
        dHdp = jax.grad(self.hamiltonian, argnums=1)(q, p)   # [N] = q_dot
        dHdq = jax.grad(self.hamiltonian, argnums=0)(q, p)   # [N] = conservative force

        # p_dot = -dH/dq + A(q)u - D(q)q_dot
        p_dot = -dHdq + A @ u - D @ q_dot       # [N]

        # Convert p_dot back to q_ddot
        # p = M q_dot  =>  p_dot = M q_ddot + (dM/dt) q_dot
        # For simplicity here: q_ddot ≈ M⁻¹ p_dot
        # (exact when M is constant; for variable M add Coriolis correction)
        q_ddot = jnp.linalg.solve(M, p_dot)     # [N]

        return q_ddot
    
# rk4 integrator
def rk4(model, q, q_dot, u, dt):
    q_ddot = model(q,q_dot,u)
    

