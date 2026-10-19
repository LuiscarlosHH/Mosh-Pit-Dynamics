import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N=1000
L=20
R_0=4
sigma_v=1
V0=10
sigma_f=3
dt=0.05

rng = np.random.default_rng()

positions=rng.uniform(-L/2,L/2,size=(N,2))
velocities = rng.normal(0, sigma_v, size=(N, 2))

def central_force(positions, V0, sigma):
    r_squared = np.sum(positions**2, axis=1)
    factors = V0 / sigma**2 * np.exp(-r_squared / (2 * sigma**2))
    forces = factors[:, None] * positions
    return forces

def update(positions, velocities, V0, sigma, dt):
    forces = central_force(positions, V0, sigma)
    accelerations = forces
    new_positions = positions + velocities * dt + accelerations/2*dt**2
    forces = central_force(new_positions, V0, sigma)
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
    update(positions, velocities, V0, sigma_f, dt)

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

