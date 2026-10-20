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

### Interparticle Interactions

Particles interact through a short-range repulsive force, modeled as a linear elastic contact.

For two particles \(i\) and \(j\), the separation vector and distance are defined as:

\[
\mathbf d_{ij}=\mathbf r_i-\mathbf r_j,
\qquad
d_{ij}=|\mathbf d_{ij}|
\]

The interaction force is:

\[
\mathbf F_{ij}=
\begin{cases}
k(d_0-d_{ij})\dfrac{\mathbf d_{ij}}{d_{ij}}, & 0<d_{ij}<d_0\\
0, & \text{otherwise}
\end{cases}
\]

where:

- \(k\) is the interaction stiffness.
- \(d_0\) is the effective contact distance.

The total force acting on each particle is:

\[
\mathbf F_i=\mathbf F_{\mathrm{central},i}+\sum_{j\ne i}\mathbf F_{ij}
\]

Pairwise interactions are computed using NumPy broadcasting, allowing all particle separations and forces to be evaluated without explicit Python loops.

The resulting forces are integrated using the Velocity Verlet method.

### Observations

Short-range interactions visibly modify particle trajectories when particles approach one another. Unlike the initial model, particles no longer move independently through the crowd.

### Current Limitations

The particles are modeled as soft objects, meaning that some overlap remains possible. Contact forces are purely repulsive and do not yet include dissipative effects or tangential friction.

The current implementation evaluates all particle pairs, resulting in \(O(N^2)\) computational complexity.

### Future Improvements

- Introduce different interaction parameters for active participants and spectators.
- Investigate numerical stability and energy conservation.
- Optimize interaction calculations for larger particle populations.

### Next Steps

- Implement short-range interparticle repulsion.
- Introduce damping and active motion.
- Develop tangential motion around the central region.
- Simulate transitions between different crowd behaviors.
- Analyze physical quantities such as energy and particle density.

## Technologies

Python, NumPy, Matplotlib.