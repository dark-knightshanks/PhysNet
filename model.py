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
    k1x = q_dot
    k1v = model(q, q_dot, u)

    k2x = q_dot + 0.5*dt*k1v
    k2v = model(
            q + 0.5*dt*k1x,
            q_dot + 0.5*dt*k1v)

    k3x = q_dot + 0.5*dt*k2v
    k3v = model(
            q + 0.5*dt*k2x,
            q_dot + 0.5*dt*k2v)

    k4x = q_dot + dt*k3v
    k4v = model(
            q + dt*k3x,
            q_dot + dt*k3v)

    x_new = q + dt/6 * (
            k1x +
            2*k2x +
            2*k3x +
            k4x)

    v_new = q_dot + dt/6 * (
            k1v +
            2*k2v +
            2*k3v +
            k4v)

    return x_new, v_new
    
def trajectory_rollout(model, q0, q_dot0, u_seq, dt):
    q,q_dot = q0, q_dot0
    q_traj = [q0]
    q_dot_traj = [q_dot0]

    for u_t in u_seq:
        q,q_dot = rk4(model, q, q_dot, u_t, dt)
        q_traj.append(q)
        q_dot_traj.append(q_dot)
    return jnp.stack(q_traj), jnp.stack(q_dot_traj) # stacls on a new axis 

def acceleration_loss(model, batch):
    # vmap vectorises the model call over the batch dimension
    q_ddot_pred = jax.vmap(model)(
        batch['q'],
        batch['q_dot'],
        batch['u']
    )                                           # [B, N]
    return jnp.mean((q_ddot_pred - batch['q_ddot_gt']) ** 2) # processes more data in given batches/ time intervals


def trajectory_loss(model, q0, q_dot0, u_seq, q_traj_gt, dt):
    q_traj_pred, _ = trajectory_rollout(model, q0, q_dot0, u_seq, dt)
    return jnp.mean((q_traj_pred - q_traj_gt) ** 2)


def combined_loss(model, batch, q0, q_dot0, u_seq, q_traj_gt, dt,
                  alpha=1.0, beta=0.1):
    acc_loss  = acceleration_loss(model, batch)
    traj_loss = trajectory_loss(model, q0, q_dot0, u_seq, q_traj_gt, dt)
    return alpha * acc_loss + beta * traj_loss








