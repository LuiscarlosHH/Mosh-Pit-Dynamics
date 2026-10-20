import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

N=200
p=0.7
R_b = 12.0
k_w = 10.0
R_0=4
sigma_v=1
V0_E=25
V0_P=10
gamma_P = 0.1
gamma_E = 2.0
k_EE=8
k_PE=5
k_PP=3
sigma_f = 8
d0 = 0.5
alpha=1.0
beta=0.5
R_a=1.5
lambda_align=1
dt = 0.05
t_final = 100.0
n_steps = round(t_final / dt)

def participant_potential(t):
    if t < 50:
        return 10.0
    elif t < 70:
        return -10.0
    else:
        return 10.0

def alignment_strength(t):
    if 50 <= t < 70:
        return 0.0
    return lambda_align

def initialize(seed=None):
    rng = np.random.default_rng(seed)

    N_P = round(p * N)
    indices = rng.choice(N, size=N_P, replace=False)

    types = np.zeros(N, dtype=int)
    types[indices] = 1

    K = np.array([k_EE, k_PE, k_PP])[
        types[:, None] + types[None, :]
    ]

    V0_particles = np.array([V0_E, V0_P], dtype=float)[types]
    gamma_particles = np.array([gamma_E, gamma_P])[types]

    u = rng.random(N)
    theta = rng.uniform(0, 2*np.pi, N)
    r = R_b * np.sqrt(u)

    positions = r[:, None] * np.column_stack((
        np.cos(theta), np.sin(theta)
    ))

    velocities = rng.normal(0, sigma_v, size=(N, 2))

    return positions, velocities, types, K, V0_particles, gamma_particles


def wall_force(positions, R_b, k_w):
    r = np.linalg.norm(positions, axis=1)
    limit = np.maximum(0, r - R_b)
    safe_r = np.where(r > 0, r, 1.0)    
    forces = -k_w*(limit/safe_r)[:, None]*positions
    return forces

def central_force(positions, V0_particles, sigma):
    r_squared = np.sum(positions**2, axis=1)
    factors = V0_particles / sigma**2 * np.exp(-r_squared / (2 * sigma**2))
    forces = factors[:, None] * positions
    return forces

def interaction_force(positions, K, d0):
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    mask = (distances < d0) & (distances > 0)
    safe_distances = np.where(mask, distances, 1.0)
    factors = mask * K * (d0-distances)/safe_distances
    pair_forces = factors[:, :, None] * differences
    forces = np.sum(pair_forces,axis=1)
    return forces

def total_force(positions, R_b, k_w, V0, sigma, k, d0):
    return (
        central_force(positions, V0, sigma)
        + interaction_force(positions, k, d0)
        + wall_force(positions, R_b, k_w)
    )

def alignment_force(positions, velocities, types, R_a, lambda_align):
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    mask_participants = (types[:, None] == 1) & (types[None, :] == 1)
    mask = (distances < R_a) & (distances > 0) & mask_participants
    n_neighbors = np.sum(mask, axis=1)
    velocity_sum = mask @ velocities
    velocity_mean = velocity_sum / np.maximum(n_neighbors[:, None], 1)
    forces = lambda_align * (velocity_mean - velocities)
    forces[n_neighbors == 0] = 0
    return forces


def update_active_velocities(
    positions, velocities, types, gamma_particles,
    alpha, beta, R_a, lambda_align, dt
):
    speed_squared = np.sum(velocities**2, axis=1)

    factors = alpha * types - gamma_particles - beta * types * speed_squared

    forces = factors[:, None] * velocities
    forces += alignment_force(positions, velocities, types, R_a, lambda_align)

    velocities += dt * forces  

def update(positions, R_b, k_w,  velocities, types, gamma_particles, V0, sigma, k, d0, alpha, beta, R_a, lambda_align, dt):
    update_active_velocities(positions, velocities, types, gamma_particles, alpha, beta, R_a, lambda_align, dt/2)
    accelerations = total_force(positions, R_b, k_w, V0,sigma,k,d0)
    new_positions = positions + velocities * dt + accelerations/2*dt**2
    forces = total_force(new_positions, R_b, k_w, V0,sigma,k,d0)
    accelerations2 = forces
    velocities += (accelerations+accelerations2)/2*dt

    update_active_velocities(new_positions, velocities, types, gamma_particles, alpha, beta, R_a, lambda_align, dt/2)

    positions[:] = new_positions

def rotational_order(positions, velocities, types):
    mask = types == 1
    positions_P = positions[mask]
    velocities_P = velocities[mask]
    vec_prod = (
    positions_P[:, 0] * velocities_P[:, 1]
    - positions_P[:, 1] * velocities_P[:, 0])
    r = np.linalg.norm(positions_P, axis=1)
    v = np.linalg.norm(velocities_P, axis=1)
    denominator = r * v
    phi = np.zeros_like(denominator)
    np.divide(vec_prod, denominator, out=phi, where=denominator > 0)
    rot_ord = np.mean(phi)
    return rot_ord

def radial_movement(positions, velocities, types):
    mask = types == 1
    positions_P = positions[mask]
    velocities_P = velocities[mask]
    esc_prod = (
    positions_P[:, 0] * velocities_P[:, 0]
    + positions_P[:, 1] * velocities_P[:, 1])
    r = np.linalg.norm(positions_P, axis=1)
    v = np.linalg.norm(velocities_P, axis=1)
    denominator = r * v
    psi = np.zeros_like(denominator)
    np.divide(esc_prod, denominator, out=psi, where=denominator > 0)
    rad_mov = np.mean(psi)
    return rad_mov


def radial_analysis(positions, types):
    r = np.linalg.norm(positions, axis=1)
    mask_p = types == 1
    r_p = r[mask_p]
    mask_e = types == 0
    r_e = r[mask_e]
    mean_r_P = np.mean(r_p)
    mean_r_E = np.mean(r_e)
    return mean_r_P, mean_r_E


def run_analysis(seeds=range(5), time_dependent_potential=False):
    """Por defecto reproduce el análisis original con V0_P constante.

    Usa time_dependent_potential=True para aplicar el mismo protocolo temporal
    que en la animación. El potencial se evalúa al inicio de cada paso.
    """
    seeds = tuple(seeds)
    times = np.arange(n_steps + 1) * dt
    fig, ax = plt.subplots()
    fig_radial, ax_radial = plt.subplots()
    fig_movement, ax_movement = plt.subplots()
    results = {}

    for seed in seeds:
        positions, velocities, types, K, V0_particles, gamma_particles = initialize(seed)
        history = [rotational_order(positions, velocities, types)]
        mean_r_P, mean_r_E = radial_analysis(positions, types)
        radial_P_history = [mean_r_P]
        radial_E_history = [mean_r_E]
        radial_movement_history = [radial_movement(positions, velocities, types)]

        for step in range(n_steps):
            t = step * dt

            if time_dependent_potential:
                V0_particles[types == 1] = participant_potential(t)

            update(
                positions, R_b, k_w, velocities, types,
                gamma_particles, V0_particles, sigma_f,
                K, d0, alpha, beta, R_a,
                alignment_strength(t) if time_dependent_potential else lambda_align,
                dt
            )
            history.append(rotational_order(positions, velocities, types))
            mean_r_P, mean_r_E = radial_analysis(positions, types)
            radial_P_history.append(mean_r_P)
            radial_E_history.append(mean_r_E)
            radial_movement_history.append(radial_movement(positions, velocities, types))

        results[seed] = {
            "rotational_order": np.array(history),
            "mean_radius_participants": np.array(radial_P_history),
            "mean_radius_spectators": np.array(radial_E_history),
            "radial_movement": np.array(radial_movement_history)
        }
        ax.plot(times, history, label=f"Semilla {seed}")
        ax_radial.plot(times, radial_P_history, label=f"Participantes — semilla {seed}")
        ax_radial.plot(times, radial_E_history, linestyle="--", label=f"Espectadores — semilla {seed}")
        ax_movement.plot(times, radial_movement_history,label=f"Semilla {seed}")

    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Tiempo de simulación")
    ax.set_ylabel("Orden rotacional Φ")
    ax.set_title(f"Orden rotacional — {len(seeds)} simulaciones")
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(True)

    ax_radial.axhline(R_b, color="black", linestyle=":", label="R_b")
    ax_radial.axhline(2 * R_b / 3, color="gray", linestyle=":", label="Media uniforme")
    ax_radial.set_xlabel("Tiempo de simulación")
    ax_radial.set_ylabel("Radio medio")
    ax_radial.set_title("Evolución radial de participantes y espectadores")
    ax_radial.legend()
    ax_radial.grid(True)

    ax_movement.axhline(0, color="gray", linestyle="--")
    ax_movement.axvspan(50, 70, color="orange", alpha=0.15)

    ax_movement.set_xlabel("Tiempo de simulación")
    ax_movement.set_ylabel("Movimiento radial Ψ")
    ax_movement.set_title("Movimiento radial — 5 simulaciones")
    ax_movement.set_ylim(-1, 1)
    ax_movement.legend()
    ax_movement.grid(True)
    plt.show()

    return times, results



def run_animation(seed=0):
    """Anima una simulación independiente con V0_P dependiente del tiempo."""

    positions, velocities, types, K, V0_particles, gamma_particles = initialize(seed)

    steps_per_frame = 5
    n_frames = (n_steps + steps_per_frame - 1) // steps_per_frame

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=types, cmap="coolwarm", vmin=0, vmax=1
    )

    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="blue", label="Espectadores"),
        Line2D([], [], marker="o", linestyle="", color="red", label="Participantes"),
    ])

    ax.set_aspect("equal")
    ax.set_xlim(-R_b - 3, R_b + 3)
    ax.set_ylim(-R_b - 3, R_b + 3)

    def init_animation():
        scatter.set_offsets(positions)
        return scatter,

    def animate(frame):
        for i in range(steps_per_frame):
            step = frame * steps_per_frame + i

            if step >= n_steps:
                break

            t = step * dt  # Tiempo físico actual

            V0_particles[types == 1] = participant_potential(t)

            update(
                positions, R_b, k_w, velocities, types,
                gamma_particles, V0_particles, sigma_f,
                K, d0, alpha, beta, R_a, alignment_strength(t), dt
            )

        scatter.set_offsets(positions)
        return scatter,

    animation = FuncAnimation(
        fig,
        animate,
        init_func=init_animation,
        frames=n_frames,
        interval=20,
        blit=True,
        repeat=False
    )

    plt.show()
    return animation

steps_per_frame = 5
n_frames = n_steps // steps_per_frame


def main():
    experiment = "analysis"  # Elige "animation" o "analysis".
    if experiment == "animation":
        return run_animation(seed=0)
    if experiment == "analysis":
        return run_analysis(time_dependent_potential=True)
    raise ValueError(f"Experimento desconocido: {experiment!r}")


if __name__ == "__main__":
    main()



