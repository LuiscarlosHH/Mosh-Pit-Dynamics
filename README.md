# Mosh Pit Dynamics

*A Particle-Based Simulation of Collective Motion*

## Overview

Mosh Pit Dynamics is a Python project exploring how particle-based models can reproduce aspects of collective motion observed in concert crowds.

The simulation begins with particles randomly distributed inside a square enclosure. A central repulsive potential causes particles to move away from the center, producing a low-density circular region.

## Current Implementation — v0.1

- Random initial particle positions and velocities.
- Two-dimensional motion inside a square enclosure.
- Reflective boundary conditions.
- Central Gaussian repulsive potential.
- Numerical integration using the Velocity Verlet method.
- Real-time visualization with Matplotlib.

### Physical Model

Each particle follows Newton's second law:

\[
m\ddot{\mathbf r}_i=\mathbf F_i
\]

The central potential is defined as:

\[
V(r)=V_0\exp\left(-\frac{r^2}{2\sigma^2}\right)
\]

The resulting force is:

\[
\mathbf F(\mathbf r)=\frac{V_0}{\sigma^2}
\exp\left(-\frac{r^2}{2\sigma^2}\right)\mathbf r
\]

For now, particle masses are set to unity.

### Observations

The repulsive potential produces a low-density region around the center. Depending on their initial conditions, particles may be reflected away from the central region, penetrate it, or follow curved trajectories around it.

### Limitations

Particles currently do not interact with one another. There is no friction, self-propulsion, or collective alignment. Wall collisions are treated approximately.

### Next Steps

- Implement short-range interparticle repulsion.
- Introduce damping and active motion.
- Develop tangential motion around the central region.
- Simulate transitions between different crowd behaviors.
- Analyze physical quantities such as energy and particle density.

## Technologies

Python, NumPy, Matplotlib.