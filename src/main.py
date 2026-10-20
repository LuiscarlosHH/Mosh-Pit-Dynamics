import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

N=500
p=0.7
L=20
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
lambda_align=0.2
dt = 0.05

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

positions=rng.uniform(-L/2,L/2,size=(N,2))
velocities = rng.normal(0, sigma_v, size=(N, 2))

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

def total_force(positions, V0, sigma, k, d0):
    return (
        central_force(positions, V0, sigma)
        + interaction_force(positions, k, d0)
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

def update(positions, velocities, types, gamma_particles, V0, sigma, k, d0, alpha, beta, R_a, lambda_align, dt):
    update_active_velocities(positions, velocities, types, gamma_particles, alpha, beta, R_a, lambda_align, dt/2)
    accelerations = total_force(positions,V0,sigma,k,d0)
    new_positions = positions + velocities * dt + accelerations/2*dt**2
    forces = total_force(new_positions,V0,sigma,k,d0)
    accelerations2 = forces
    velocities += (accelerations+accelerations2)/2*dt

    # X axis
    mask_xp = new_positions[:, 0] >= L/2
    mask_xn = new_positions[:, 0] <= -L/2

    new_positions[mask_xp, 0] = L - new_positions[mask_xp, 0]
    new_positions[mask_xn, 0] = -L - new_positions[mask_xn, 0]

    velocities[mask_xp | mask_xn, 0] *= -1

    # Y axis
    mask_yp = new_positions[:, 1] >= L/2
    mask_yn = new_positions[:, 1] <= -L/2

    new_positions[mask_yp, 1] = L - new_positions[mask_yp, 1]
    new_positions[mask_yn, 1] = -L - new_positions[mask_yn, 1]

    velocities[mask_yp | mask_yn, 1] *= -1

    update_active_velocities(positions, velocities, types, gamma_particles, alpha, beta, R_a, lambda_align, dt/2)

    positions[:] = new_positions


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

def animate(frame):
    update(positions, velocities, types, gamma_particles, V0_particles, sigma_f, K, d0, alpha, beta, R_a, lambda_align, dt)

    scatter.set_offsets(positions)

    return scatter,

animation = FuncAnimation(
    fig,
    animate,
    frames=500,
    interval=60,
    blit=True
)

plt.show()

