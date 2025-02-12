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
from matplotlib.widgets import Slider

def attention_mixture(
    x: Float[Array, "D"],
    mix_weights: Float[Array, "N"],
    means: Float[Array, "N D"],
    cov_mats: Float[Array, "N D D"]
) -> Float[Array, "N"]:
    """Computes numerically stable attention weights for a mixture of Gaussians."""
    # Compute log of the mixture weights (add epsilon to avoid log(0))
    log_mix_weights = jnp.log(mix_weights + 1e-12)
    # Compute log-density for each Gaussian component
    log_gauss = vmap(jst.multivariate_normal.logpdf, in_axes=(None, 0, 0))(x, means, cov_mats)
    # Sum to get the log of the weighted densities
    log_densities = log_mix_weights + log_gauss
    # For numerical stability subtract the maximum log value (log-sum-exp trick)
    max_log = jnp.max(log_densities)
    exp_shifted = jnp.exp(log_densities - max_log)
    # Compute the normalized attention weights
    attention = exp_shifted / jnp.sum(exp_shifted)
    
    return attention

def score_mixture(
    x: Float[Array, "D"],
    mix_weights: Float[Array, "N"],
    means: Float[Array, "N D"],
    cov_mats: Float[Array, "N D D"]
) -> Float[Array, "D"]:
    """
    Computes the score of a mixture of Gaussians:
      score(x) = sum_i w_i(x) * Sigma_i^{-1} (mu_i - x)
    where w_i(x) = (pi_i * N(x; mu_i, Sigma_i)) / (sum_j pi_j * N(x; mu_j, Sigma_j))
    """
    # Get attention weights for each component
    attention = attention_mixture(x, mix_weights, means, cov_mats)
    # Compute inverse covariance for each component
    inv_covs = vmap(jnp.linalg.inv)(cov_mats)
    # Broadcast x to have the same shape as means: (N, D)
    diff = means - repeat(x, "... D -> ... N D", N=means.shape[0])
    # Compute per-component contribution to the score
    # This gives a (N, D) tensor for each component.
    per_component_score = vmap(lambda inv_cov, d: inv_cov @ d)(inv_covs, diff)
    # Weight by the attention and sum over components:
    score = jnp.sum(attention[:, None] * per_component_score, axis=0)
    return score


def score_mixture_a(
    x: Float[Array, "D"],
    mix_weights: Float[Array, "N"],
    means: Float[Array, "N D"],
    cov_mats: Float[Array, "N D D"]
) -> Float[Array, "N"]:
    """
    Computes the score of a mixture of Gaussians:
      score(x) = sum_i w_i(x) * Sigma_i^{-1} (mu_i - x)
    where w_i(x) = (pi_i * N(x; mu_i, Sigma_i)) / (sum_j pi_j * N(x; mu_j, Sigma_j))
    """
    N,D = means.shape
    # Attention @ P @ (mu - x)
    precision = jnp.linalg.inv(cov_mats)
    attention = attention_mixture(x, mix_weights, means, cov_mats)
    difference = means - repeat(x, "... D -> ... N D", N=N)
    score = attention @ vmap(jnp.matmul)(precision, difference) 

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


def plot_basins(solutions: Float[Array, "t n D"], attractors: Float[Array, "N D"]):

    n_steps, n, D = solutions.shape
    N, D = attractors.shape

    # min_range = np.min(np.min(solutions[:, :, 0]), np.min(solutions[:, :, 1]))  
    # max_range = np.max(np.max(solutions[:, :, 0]), np.max(solutions[:, :, 1]))

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    ax.set_aspect('equal')
    # ax.set_xlim((min_range, max_range))
    # ax.set_ylim((min_range, max_range))
    
    # Plot attractors
    ax.scatter(attractors[:, 0], attractors[:, 1], 
               marker='*', c='red', s=150, zorder=0)
    # Map points to attractors
    distance_to_attractors = jnp.linalg.norm(solutions[-1, :, None, :] - attractors[None, :, :], axis=-1)
    inx_closest_attractor = jnp.argmin(distance_to_attractors, axis=1)

    # Generate unique colors for each attractor
    unique_attractors = np.unique(inx_closest_attractor)
    colors = np.array(plt.cm.jet(jnp.linspace(0, 1, len(unique_attractors))))
    color_map = {attr: colors[i] for i, attr in enumerate(unique_attractors)}  # Map attractor index to color

    # Initial plot (t = 0)
    scatter = ax.scatter(solutions[0, :, 0], solutions[0, :, 1], 
                         c=[color_map[int(attr)] for attr in inx_closest_attractor],
                         zorder=1, alpha=float(len(unique_attractors) * 10 / solutions.shape[1]),
                         edgecolors='none')


    # Make sliders
    ax_time = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    allowed_time_index = jnp.arange(n_steps)
    slider_time = Slider(
            ax_time, "Time Index", 0, n_steps,
            valinit=0, valstep=allowed_time_index, 
            color="green"
    )

    def update(t):
        t = int(slider_time.val)
        scatter.set_offsets(solutions[t])  # Update positions
        scatter.set_color([color_map[int(attr)] for attr in inx_closest_attractor])  # Reapply colors
        fig.canvas.draw_idle()

    slider_time.on_changed(update)
    plt.show()
    

def main(debug=True):
    means = jnp.array([
        [-3, 0], 
        [3, 0],
        [0, -3],
        [0, 3],
    ])
    N, D = means.shape
    covs = jnp.ones(N) * 0.01
    mix_weights = jnp.ones(N) / N
    cov_mats = covs[:, None, None,] * jnp.eye(D)

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
    num_steps = 32
    num_samples = 1000
    key = jr.PRNGKey(0)

    # x_0 = jr.multivariate_normal(key, jnp.zeros(2), jnp.eye(2), shape=(1000, 1))
    step_indices = jnp.arange(num_steps)
    T = 1.0
    x_0 = jr.multivariate_normal(key, jnp.zeros(2), T * jnp.eye(2), shape=1000)
    t_steps = jnp.linspace(T, 0.0, num_steps)
    
    def score_match(t, y, args):
        cov_adjusted = cov_mats + t**2 * jnp.eye(D)
        return -t * vmap(lambda x: score_mixture_a(x, mix_weights, means, cov_adjusted))(y)

    # Initialize solver
    jax.config.update("jax_debug_nans", True)
    with jax.disable_jit():
        sol = diffrax.diffeqsolve(
                terms = diffrax.ODETerm(score_match),
                solver = diffrax.Euler(),
                y0 = x_0,
                t0 = T,
                t1 = 0.0,
                dt0 = -1/num_steps,
                saveat = diffrax.SaveAt(ts=t_steps)
        )


    assert jnp.all(sol.ys[0] == x_0)

    plot_basins(sol.ys, means)
    

if __name__ ==  "__main__":
    main(debug=False)


