import jax
import jax.numpy as jnp
import jax.scipy.stats as jst
import matplotlib.pyplot as plt
import numpy as np
import jax.random as jr 
import diffrax

from jax import vmap
from jaxtyping import Float, Array
from einops import repeat, rearrange
from functools import partial


def attention_mixture(
    x: Float[Array, "D"],
    mix_weights: Float[Array, "N"],
    means: Float[Array, "N D"],
    cov_mats: Float[Array, "N D D"]
) -> Float[Array, "N"]:
    """ Computes attention weights for a mixture of Gaussians. """

    N, D = means.shape
    densities =  vmap(jst.multivariate_normal.pdf, in_axes=(None, 0, 0))(x, means, cov_mats)
    densities = mix_weights * densities
    attention = densities.T / jnp.sum(densities)
    return attention


def score_mixture(
    x: Float[Array, "D"],
    mix_weights: Float[Array, "N"],
    means: Float[Array, "N D"],
    cov_mats: Float[Array, "N D D"]
) -> Float[Array, "N"]:

    N,D = means.shape

    attention = attention_mixture(x, mix_weights, means, cov_mats)
    difference = means - repeat(x, "... D -> ... N D", N=N)
    score = attention @ difference

    return score

def pdf_mixture(
    x: Float[Array, "1 D"],
    mix_weights: Float[Array, "1 N"],
    means: Float[Array, "N D"],
    cov_mats: Float[Array, "N D D"]
) -> Float[Array, " 1 N"]:

    # Weighted sum over the density of mixstures
    densities = mix_weights @ vmap(jst.multivariate_normal.pdf, in_axes=(None, 0, 0))(x, means, cov_mats)
    return densities

def plot_density_map_pdf(pdf):
    """ """

    # Generate a grid of points
    x_range = jnp.linspace(-3, 3, 100)
    y_range = jnp.linspace(-3, 3, 100)
    X, Y = jnp.meshgrid(x_range, y_range)

    # Compute density values
    points = rearrange([X, Y], "d h w -> (h w) 1 d" )
    Z = pdf(points)
    Z = Z.reshape(X.shape)

    # Plot the density map
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, Z, levels=50, cmap="plasma")
    plt.colorbar(label="Density")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Density")
    plt.show()

def plot_quiver(gradient):
    # Generate grid of points
    resolution = 100
    X, Y = jnp.meshgrid(jnp.linspace(-3, 3, resolution), jnp.linspace(-3, 3, resolution))
    # Compute gradient
    points = rearrange([X, Y], "d h w -> (h w) 1 d" )
    U, V = rearrange(gradient(points) , "(h w) 1 D -> D h w", h=resolution)

    fig1, ax1 = plt.subplots()
    ax1.set_title('Arrows scale with plot width, not view')
    Q = ax1.quiver(X, Y, U, V, units='width')
    qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
    # ax1.streamplot(np.array(X), np.array(Y), np.array(U), np.array(V))
    plt.show()


def main(debug=True):
    # Generate attention
    mix_weights = jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
    means = jnp.array([[0, 0], [-1, 1], [1, -1], [1, 1], [-1, -1]])
    covs = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
    cov_mats = covs[:, None, None,] * jnp.eye(2)

    if debug:
        print(f"mix_weights\t: {mix_weights.shape}")
        print(f"means\t\t: {means.shape}")
        print(f"cov_mats\t: {cov_mats.shape}")

    assert jnp.isclose(jnp.sum(mix_weights), 1.0), \
        f"Mixture weights should close to 1.0, got {mix_weights} which sums to {jnp.sum(mix_weights)}"
    # assert rank of tensors are correct
    assert len(means.shape) == 2
    assert len(cov_mats.shape) == 3, f"Rank of cov_mats should be 3, got {cov_mats.shape} with len = {len(cov_mats.shape)}"
    assert len(mix_weights.shape) == 1
    
    
    # Simulate diffusion 
    num_steps = 16
    num_samples = 1000
    key = jr.PRNGKey(0)

    # x_0 = jr.multivariate_normal(key, jnp.zeros(2), jnp.eye(2), shape=(1000, 1))
    x_0 = jr.multivariate_normal(key, jnp.zeros(2), jnp.eye(2), shape=1000)
    step_indices = jnp.arange(num_steps)
    t_steps = jnp.linspace(1, 0, num_steps)
    
    def score_match(t, y, args):
        return -t * vmap(lambda x: score_mixture(x, mix_weights, means, cov_mats + t**2))(y)

    # Initialize solver
    sol = diffrax.diffeqsolve(
            terms = diffrax.ODETerm(score_match),
            solver = diffrax.Euler(),
            y0 = x_0,
            t0 = 1.0,
            t1 = 0.0,
            dt0 = -1/num_steps,
            saveat = diffrax.SaveAt(ts=t_steps)
        )

    print(sol.ys.shape)
    plt.plot(sol.ys[0])
    plt.show()


    

if __name__ ==  "__main__":
    main(debug=False)


