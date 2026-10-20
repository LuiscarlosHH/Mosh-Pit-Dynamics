import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N=200
L=20
R_0=4
sigma_v=1
V0=10
sigma_f=3
k = 5
d0 = 0.5
dt = 0.05

rng = np.random.default_rng()

positions=rng.uniform(-L/2,L/2,size=(N,2))
velocities = rng.normal(0, sigma_v, size=(N, 2))

def central_force(positions, V0, sigma):
    r_squared = np.sum(positions**2, axis=1)
    factors = V0 / sigma**2 * np.exp(-r_squared / (2 * sigma**2))
    forces = factors[:, None] * positions
    return forces

def interaction_force(positions, k, d0):
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    mask = (distances < d0) & (distances > 0)
    safe_distances = np.where(mask, distances, 1.0)
    factors = mask * k * (d0-distances)/safe_distances
    pair_forces = factors[:, :, None] * differences
    forces = np.sum(pair_forces,axis=1)
    return forces

def total_force(positions, V0, sigma, k, d0):
    return (
        central_force(positions, V0, sigma)
        + interaction_force(positions, k, d0)
    )

def update(positions, velocities, V0, sigma, k, d0, dt):
    forces = total_force(positions,V0,sigma,k,d0)
    accelerations = forces
    new_positions = positions + velocities * dt + accelerations/2*dt**2
    forces = total_force(positions,V0,sigma,k,d0)
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

    positions[:] = new_positions


fig, ax = plt.subplots()

scatter = ax.scatter(positions[:, 0], positions[:, 1])

def animate(frame):
    update(positions, velocities, V0, sigma_f, k, d0, dt)

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

