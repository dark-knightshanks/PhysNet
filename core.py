import jax
import jax.numpy as jnp
from flax import nnx

jnp.random.seed(69)


# Mass Network (M-NN)
# Input : q(pos) with shape N 
# Output : Lower Triangular Matrix elements [(N^2+N)/2]
class MNN(nnx.Module):
    def __init__(self, N, hidden_dims, rngs: nnx.Rngs):
        self.N = N
        self.output = (N * N + N) // 2
        layers = []
        input_dim = N
        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim
        layers.append(nnx.Linear(input_dim, self.output, rngs=rngs))
        self.layers = layers

    def __call__(self, q):
        x = q
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = jax.nn.relu(x)
        x = self.layers[-1](x)
        
        diagonal = x[:self.N]
        lower_left = x[self.N:]
        epsilon = 1e-3
        diagonal = jax.nn.softplus(diagonal) + epsilon
        L_M = self._build_lower_triangular(diagonal, lower_left)
        # Mass matrix via Cholesky: M = L_M @ L_M^T
        M = L_M @ L_M.T
        
        return M
    
    def _build_lower_triangular(self, diagonal, lower_left):
        N = self.N
        L_N = jnp.zeros((N, N))
        
        # Set diagonal
        L_N = L_N.at[jnp.arange(N), jnp.arange(N)].set(diagonal)
        
        # Fill lower triangle
        idx = 0
        for i in range(1, N):
            for j in range(i):
                L_N = L_N.at[i, j].set(lower_left[idx])
                idx += 1
        
        return L_N


# Potential Energy Network
# Input: q(pos) [N]
# Output: Scalar value
class PNN(nnx.Module):
    def __init__(self, N, hidden_dims, rngs: nnx.Rngs):
        self.N = N
        self.output = 1
        layers = []
        input_dim = N
        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim
        layers.append(nnx.Linear(input_dim, self.output, rngs=rngs))
        self.layers = layers
    
    def __call__(self, q):
        x = q
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = jax.nn.relu(x)
        # Final layer (no activation)
        x = self.layers[-1](x)
        return jnp.squeeze(x)


# Damping Network
# Input : q(pos) with shape N 
# Output: Lower Triangular Matrix elements [(N^2+N)/2]
class DNN(nnx.Module):
    def __init__(self, N, hidden_dims, rngs: nnx.Rngs):
        self.N = N
        self.output = (N * N + N) // 2
        layers = []
        input_dim = N
        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim
        layers.append(nnx.Linear(input_dim, self.output, rngs=rngs))
        self.layers = layers
    
    def __call__(self, q):
        x = q
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = jax.nn.relu(x)
        x = self.layers[-1](x)
        
        diagonal = x[:self.N]
        lower_left = x[self.N:]
        diagonal = jax.nn.softplus(diagonal)
        D_N = self._build_lower_triangular(diagonal, lower_left)
        
        return D_N @ D_N.T
    
    def _build_lower_triangular(self, diagonal, lower_left):
        N = self.N
        D = jnp.zeros((N, N))
        
        # Set diagonal
        D = D.at[jnp.arange(N), jnp.arange(N)].set(diagonal)
        
        # Fill lower triangle
        idx = 0
        for i in range(1, N):
            for j in range(i):
                D = D.at[i, j].set(lower_left[idx])
                idx += 1
        
        return D


# Input Matrix Network
# Input : q(pos) with shape N 
# Output: NxM  M = size of control vector
class ANN(nnx.Module):
    def __init__(self, N, M, hidden_dims, rngs: nnx.Rngs, use_sigmoid=True):
        self.N = N
        self.M = M
        self.output = N * M
        self.use_sigmoid = use_sigmoid
        layers = []
        input_dim = N
        
        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            input_dim = hidden_dim
        # Final layer outputs N*M values
        layers.append(nnx.Linear(input_dim, self.output, rngs=rngs))
        self.layers = layers
    
    def __call__(self, q):
        x = q
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = jax.nn.relu(x)
        x = self.layers[-1](x)
        
        if self.use_sigmoid:
            x = jax.nn.sigmoid(x)
        
        A = x.reshape(self.N, self.M)
        return A
