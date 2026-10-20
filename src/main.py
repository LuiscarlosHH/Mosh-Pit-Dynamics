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
sigma_f=3
d0 = 0.5
alpha=1.0
beta=0.5
R_a=1.5
lambda_align=1
dt = 0.05
t_final = 100.0
n_steps = round(t_final / dt)

rng = np.random.default_rng()

N_P = round(p * N)
indices = rng.choice(N, size=N_P, replace=False)
types = np.zeros(N, dtype=int)
types[indices] = 1
mask_types = types[:, None] + types [None, :]

k_values = np.array([k_EE, k_PE, k_PP])
K = k_values[mask_types]

V0_values = np.array([V0_E, V0_P])
V0_particles = V0_values[types]

gamma_particles = np.array([gamma_E, gamma_P])[types]

u = rng.random(N)
theta = rng.uniform(0, 2*np.pi, N)
r = R_b * np.sqrt(u)
positions = r[:,None]*np.column_stack((np.cos(theta),np.sin(theta)))
velocities = rng.normal(0, sigma_v, size=(N, 2))

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

rotational_history = []


fig, ax = plt.subplots()

scatter = ax.scatter(
    positions[:, 0],
    positions[:, 1],
    c=types,
    cmap="coolwarm",
    vmin=0,
    vmax=1
)

ax.legend(handles=[
    Line2D([], [], marker='o', linestyle='', color='blue', label='Espectadores'),
    Line2D([], [], marker='o', linestyle='', color='red', label='Participantes')
])

ax.set_aspect("equal")
ax.set_xlim(-R_b - 3, R_b + 3)
ax.set_ylim(-R_b - 3, R_b + 3)

def animate(frame):
    update(positions, R_b, k_w, velocities, types, gamma_particles, V0_particles, sigma_f, K, d0, alpha, beta, R_a, lambda_align, dt)
    rotational_history.append(rotational_order(positions, velocities, types))

    scatter.set_offsets(positions)

    return scatter

def initialize(seed):
    rng = np.random.default_rng(seed)

    N_P = round(p * N)
    indices = rng.choice(N, size=N_P, replace=False)

    types = np.zeros(N, dtype=int)
    types[indices] = 1

    K = np.array([k_EE, k_PE, k_PP])[
        types[:, None] + types[None, :]
    ]

    V0_particles = np.array([V0_E, V0_P])[types]
    gamma_particles = np.array([gamma_E, gamma_P])[types]

    u = rng.random(N)
    theta = rng.uniform(0, 2*np.pi, N)
    r = R_b * np.sqrt(u)

    positions = r[:, None] * np.column_stack((
        np.cos(theta), np.sin(theta)
    ))

    velocities = rng.normal(0, sigma_v, size=(N, 2))

    return positions, velocities, types, K, V0_particles, gamma_particles

t_final = 100.0
n_steps = round(t_final / dt)

times = np.arange(n_steps + 1) * dt

fig, ax = plt.subplots()

for seed in range(5):
    positions, velocities, types, K, V0_particles, gamma_particles = initialize(seed)

    history = [rotational_order(positions, velocities, types)]

    for step in range(n_steps):
        update(
            positions, R_b, k_w, velocities, types,
            gamma_particles, V0_particles, sigma_f,
            K, d0, alpha, beta, R_a, lambda_align, dt
        )

        history.append(rotational_order(positions, velocities, types))

    ax.plot(times, history, label=f"Semilla {seed}")

ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Tiempo de simulación")
ax.set_ylabel("Orden rotacional Φ")
ax.set_title("Orden rotacional — 5 simulaciones")
ax.set_ylim(-1, 1)
ax.legend()
ax.grid(True)

plt.show()

animation = FuncAnimation(
    fig,
    animate,
    frames=n_steps,
    interval=60,
    blit=True,
    repeat=False
)

plt.show()
