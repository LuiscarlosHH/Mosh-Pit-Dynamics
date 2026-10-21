# Mosh Pit Dynamics

An exploratory two-dimensional particle simulation of collective motion inspired by mosh pits. Active participants and damped spectators interact through soft repulsion, a central Gaussian potential, and a soft circular wall. Local velocity alignment and a prescribed temporal protocol provide a setting for investigating rotational and radial motion.

The current implementation is a simulation and analysis prototype. **Audio analysis and synchronization with music are a central future direction**, but are not implemented yet. The model uses simulation units and has not been calibrated or validated against real crowds.

## Implemented features

- Seeded initialization of two populations, uniformly distributed in a disk, with Gaussian initial velocities.
- Participant self-propulsion, speed saturation, damping, and local participant-to-participant velocity alignment.
- Population-dependent soft contact forces and central potential amplitudes.
- Constant-potential analysis or a time-dependent attraction/repulsion protocol with individual delays and attraction strengths.
- Four analysis figures: rotational order, mean radius for both populations, signed radial motion, and absolute radial motion.
- Independent animated simulation with the temporal protocol, five integration steps per frame.
- In-memory analysis arrays returned for each seed; no automatic export.

## Repository layout

```text
.
├── src/
│   └── main.py         # Parameters, forces, integration, metrics, analysis, animation
├── requirements.txt    # Direct runtime dependencies
├── README.md
├── .gitignore
└── LICENSE             # MIT license
```

## Requirements and installation

Use Python 3.11 or later as the documented baseline. The repository does not declare a formal minimum Python version or provide a version compatibility matrix.

Direct dependencies in `requirements.txt`:

- NumPy: `>=1.24,<3`
- Matplotlib: `>=3.7,<4`

A graphical Matplotlib backend is needed for interactive plots and animation. No datasets, audio files, credentials, or external services are required.

From the repository root, create an isolated environment and install the dependencies.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src/main.py
```

### Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/main.py
```

Environment activation is optional when using these interpreter paths. For the examples below, use the environment's interpreter in place of `python`, or activate the environment first.

## Running experiments

### Default analysis

```bash
python src/main.py
```

`main()` selects `experiment = "analysis"` and calls `run_analysis(time_dependent_potential=True)`. This runs seeds 0–4 for 100 simulation time units and opens four figures. Plot labels are currently in Spanish.

**Calling `run_analysis()` directly has a different default:** its `time_dependent_potential` argument defaults to `False`. This keeps the participant potential and alignment strength constant.

From the repository root:

```bash
# One seed with the temporal protocol
python -c "from src.main import run_analysis; run_analysis(seeds=[0], time_dependent_potential=True)"

# Five seeds with constant potential and alignment
python -c "from src.main import run_analysis; run_analysis()"
```

To retain the returned data, run the following in a Python session started at the repository root:

```python
from src.main import run_analysis

times, results = run_analysis(seeds=[0, 1], time_dependent_potential=True)
phi = results[0]["rotational_order"]
```

The function returns after `plt.show()` finishes (normally after closing the figure windows in a script). `times` contains the initial time and every completed integration step. With the default settings, it has 2001 entries. Each seed maps to arrays with the same length:

| Result key | Quantity |
| --- | --- |
| `rotational_order` | Participant rotational order, $\Phi$ |
| `radial_movement` | Participant signed radial motion, $\Psi$ |
| `absolute_radial_movement` | Participant absolute radial motion, $A_r$ |
| `mean_radius_participants` | Participant mean radius |
| `mean_radius_spectators` | Spectator mean radius |

### Animation

```bash
python -c "from src.main import run_animation; animation = run_animation(seed=0)"
```

Alternatively, change `experiment = "analysis"` to `experiment = "animation"` inside `main()` in `src/main.py`, then run the script. There is no command-line argument parser.

`run_animation(seed=0)` starts an independent simulation with the temporal protocol and returns the Matplotlib animation object. Keep a reference to that object in interactive sessions. The default animation has 400 frames, with five steps per frame and a requested display interval of 20 ms. Playback is not synchronized to simulation time or music. Animation does not record the analysis histories or automatically save a video.

## Model and equations

Let $\mathbf{x}_i$ and $\mathbf{v}_i$ denote particle position and velocity, $r_i=\|\mathbf{x}_i\|$, and $s_i\in\{0,1\}$ its type: spectator (0) or participant (1). All masses equal one. Radii and directional metrics are measured relative to the fixed origin, not the moving center of mass.

The implemented terms correspond to:

$$
\frac{d\mathbf{x}_i}{dt}=\mathbf{v}_i,
\qquad
\frac{d\mathbf{v}_i}{dt}=\mathbf{F}^{\mathrm{pos}}_i+\mathbf{G}_i,
$$

$$
\mathbf{F}^{\mathrm{pos}}_i=
\mathbf{F}^{\mathrm{central}}_i+
\mathbf{F}^{\mathrm{contact}}_i+
\mathbf{F}^{\mathrm{wall}}_i.
$$

### Central Gaussian potential

$$
U_i(\mathbf{x}_i,t)=V_{0,i}(t)\exp\!\left(-\frac{r_i^2}{2\sigma_f^2}\right),
$$

$$
\mathbf{F}^{\mathrm{central}}_i=-\nabla_{\mathbf{x}_i}U_i
=\frac{V_{0,i}(t)}{\sigma_f^2}
\exp\!\left(-\frac{r_i^2}{2\sigma_f^2}\right)\mathbf{x}_i.
$$

Positive amplitude repels particles from the origin; negative amplitude attracts them. The implemented Gaussian width is `sigma_f = 8`. The global `R_0 = 4` is currently unused and does not set the width or a preferred radius.

### Soft pair repulsion

For $d_{ij}=\|\mathbf{x}_i-\mathbf{x}_j\|$:

$$
\mathbf{F}^{\mathrm{contact}}_i=
\sum_{\substack{j\ne i\\0<d_{ij}<d_0}}
K_{ij}(d_0-d_{ij})\frac{\mathbf{x}_i-\mathbf{x}_j}{d_{ij}}.
$$

The stiffness is 8 for spectator/spectator pairs, 5 for mixed pairs, and 3 for participant/participant pairs. Pairs at exactly zero separation are excluded by the implementation; particles are soft and can overlap.

### Soft circular wall

$$
\mathbf{F}^{\mathrm{wall}}_i=
\begin{cases}
-k_w(r_i-R_b)\dfrac{\mathbf{x}_i}{r_i}, & r_i>R_b,\\
\mathbf{0}, & r_i\le R_b.
\end{cases}
$$

This is a restoring force outside the nominal boundary, not a hard reflection. Particles can temporarily move beyond $R_b$.

### Activity, damping, and local alignment

Define the participant neighborhood:

$$
\mathcal{N}_i=\{j:\ s_i=s_j=1,\quad 0<d_{ij}<R_a\}.
$$

$$
\mathbf{F}^{\mathrm{align}}_i=
\begin{cases}
\lambda(t)\left(\dfrac{1}{|\mathcal{N}_i|}\sum_{j\in\mathcal{N}_i}\mathbf{v}_j-\mathbf{v}_i\right), & |\mathcal{N}_i|>0,\\
\mathbf{0}, & |\mathcal{N}_i|=0,
\end{cases}
$$

$$
\mathbf{G}_i=
\left(\alpha s_i-\gamma_i-\beta s_i\|\mathbf{v}_i\|^2\right)\mathbf{v}_i
+\mathbf{F}^{\mathrm{align}}_i.
$$

Alignment uses the mean velocity, including its magnitude, rather than mean unit headings. Spectators receive linear damping and no active propulsion or alignment. There is no random forcing after initialization.

### Temporal protocol

For the time-dependent experiment, spectator amplitude remains 25. Define the base participant schedule at individual time $\tau$:

$$
q(\tau)=
\begin{cases}
10, & \tau<50,\\
-10, & 50\le\tau<70,\\
10, & \tau\ge70.
\end{cases}
$$

Each participant has a fixed sampled delay $\delta_i\sim\mathcal{U}(0,5)$ and attraction coefficient $c_i\sim\mathcal{U}(0.7,1.3)$:

$$
V_{0,i}(t)=
\begin{cases}
c_i q(t-\delta_i), & q(t-\delta_i)<0,\\
q(t-\delta_i), & q(t-\delta_i)\ge0.
\end{cases}
$$

Thus its attraction interval is $[50+\delta_i,70+\delta_i)$. Only the negative amplitude is scaled. Alignment switches globally, without individual delays:

$$
\lambda(t)=
\begin{cases}
0, & 50\le t<70,\\
\lambda_{\mathrm{align}}, & \text{otherwise}.
\end{cases}
$$

The schedule is prescribed in code; it is not extracted from audio. Constant-potential analysis uses participant amplitude 10 and alignment strength 1 throughout.

### Integration

Each `update()` applies an explicit half-step of activity/damping/alignment, a Velocity Verlet position-force update, and a second explicit half-step of activity/damping/alignment. The temporal potential and alignment coefficient are evaluated at the start of each full step and held fixed within that step.

This is not a conservative Velocity Verlet integrator for the full velocity-dependent, dissipative, time-dependent system. No convergence order or stability range has been established here.

## Default parameters and initialization

Parameters are module-level values at the top of `src/main.py`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `N`, `p` | 200, 0.7 | Total population and participant fraction; 140 participants, 60 spectators |
| `R_b`, `k_w` | 12, 10 | Nominal wall radius and wall stiffness |
| `sigma_v` | 1 | Standard deviation of each initial velocity component |
| `V0_E`, `V0_P` | 25, 10 | Spectator and baseline participant amplitudes |
| `sigma_f` | 8 | Gaussian potential width |
| `gamma_E`, `gamma_P` | 2, 0.1 | Spectator and participant damping |
| `k_EE`, `k_PE`, `k_PP` | 8, 5, 3 | Pair stiffnesses |
| `d0` | 0.5 | Contact cutoff |
| `alpha`, `beta` | 1, 0.5 | Participant propulsion and speed saturation |
| `R_a`, `lambda_align` | 1.5, 1 | Alignment radius and baseline strength |
| `dt`, `t_final` | 0.05, 100 | Integration step and duration; 2000 steps |

`initialize(seed)` uses `numpy.random.default_rng(seed)`. It randomly selects `round(p * N)` participants without replacement. Initial angles are uniform on $[0,2\pi)$ and initial radii are $R_b\sqrt{u}$ for uniform $u\in[0,1)$, giving a uniform distribution over disk area. Each initial velocity component is sampled independently from a zero-mean Gaussian with standard deviation `sigma_v`. Initial overlaps are not rejected.

When changing configuration, edit the source before running. `n_steps` is derived when the module loads; changing `dt` or `t_final` interactively requires recomputing it as well.

## Four metrics

Let $P$ denote the participant set and $N_P=|P|$. For each participant, define the normalized tangential and radial components, with a zero contribution whenever $r_i\|\mathbf{v}_i\|=0$:

$$
a_i=\begin{cases}
\dfrac{x_i v_{y,i}-y_i v_{x,i}}{r_i\|\mathbf{v}_i\|}, & r_i\|\mathbf{v}_i\|>0,\\
0, & \text{otherwise},
\end{cases}
\qquad
b_i=\begin{cases}
\dfrac{\mathbf{x}_i\cdot\mathbf{v}_i}{r_i\|\mathbf{v}_i\|}, & r_i\|\mathbf{v}_i\|>0,\\
0, & \text{otherwise}.
\end{cases}
$$

### 1. Rotational order — Φ

$$
\Phi=\frac{1}{N_P}\sum_{i\in P}a_i.
$$

Range: $[-1,1]$. Positive values indicate counterclockwise tangential motion; negative values indicate clockwise motion. Opposite rotations can cancel. A value near zero alone does not establish disorder.

### 2. Signed radial motion — Ψ

$$
\Psi=\frac{1}{N_P}\sum_{i\in P}b_i.
$$

Range: $[-1,1]$. Positive values indicate outward motion and negative values inward motion. Simultaneous inward and outward contributions can cancel.

### 3. Absolute radial motion — A_r

$$
A_r=\frac{1}{N_P}\sum_{i\in P}|b_i|.
$$

Range: $[0,1]$. This measures radial directionality without cancellation between inward and outward motion. It is the mean of absolute individual contributions, not the absolute value of $\Psi$. These three normalized metrics describe direction, not speed, flux, or kinetic energy.

### 4. Mean radius

For each population $S$ (participants or spectators):

$$
\overline{r}_S=\frac{1}{|S|}\sum_{i\in S}\|\mathbf{x}_i\|.
$$

This describes the population's spatial extent relative to the origin. The mean-radius plot includes $R_b$ and the analytical uniform-disk expectation $2R_b/3=8$ as reference lines. The latter is not an asserted simulation result. Because the wall is soft, mean radius has no strict upper bound at $R_b$.

All metrics are sampled initially and after each full step in `run_analysis()`. Empty populations are not handled explicitly: changing `p` to eliminate a group can produce undefined means (`NaN`).

## Reproducibility

- Use explicit seeds and record whether `time_dependent_potential` is enabled. Defaults are seeds 0–4 for analysis and seed 0 for animation.
- Record the source revision, all global parameters, Python/NumPy/Matplotlib versions, and platform. Dependency ranges do not lock an exact environment or guarantee bitwise agreement across versions.
- Initialization, population assignment, delays, and attraction coefficients come from the seeded generator. There is no subsequent stochastic forcing.
- The analysis function returns arrays but does not persist them, parameters, or environment metadata. Save these explicitly for a study.
- Keep the timestep, duration, initial state, and temporal protocol consistent when comparing runs. Check convergence with smaller timesteps and variability across seeds before drawing conclusions.
- For execution without a display, set a noninteractive Matplotlib backend (for example `MPLBACKEND=Agg`) before importing the module. This does not enable automatic figure or data export.

No benchmark results, experimentally validated transitions, or quantitative claims of agreement with real mosh pits are supplied by this README.

## Limitations

- Phenomenological point-particle model with unit masses and uncalibrated units; no human decision-making, body geometry, injury model, or validated crowd-safety prediction.
- Soft contacts permit overlaps; exact coincident pairs have no contact force. There is no tangential contact friction or dedicated collision dissipation term.
- Pair interactions and alignment use dense pairwise arrays with quadratic time and memory cost in population size.
- Activity, damping, alignment, and the temporal potential prevent mechanical energy conservation.
- No automated test suite is included in the current repository; timestep convergence, parameter sensitivity, and ensemble uncertainty remain to be assessed.
- Global configuration and a single implementation module; no CLI, automatic results export, checkpointing, or video export.
- Some plot titles hardcode five simulations, and the radial-motion plots shade 50–70 even for constant-potential analysis. This shading also omits individual delays; interpret the protocol from the actual arguments and equations.
- Plot labels are in Spanish. Audio loading, feature extraction, playback, and musical synchronization are not implemented.

## Roadmap

### Central future direction: audio analysis and musical synchronization

The planned audio-analysis branch is a core extension of the project. It is a development direction, not a claim that an implemented Git branch or audio pipeline currently exists.

1. Build an audio-analysis pipeline for legally usable local recordings: define channel handling, resampling, analysis windows, and normalization; extract energy envelopes, onsets, tempo, and beat timestamps.
2. Define and document how musical features control participant activity, central potential, and alignment. Replace or compare the current prescribed schedule with explicit audio-driven protocols, retaining individual response delays and heterogeneous strengths.
3. Establish a common timebase for audio features, simulation steps, visualization, and eventual playback. Specify latency, interpolation, and the relation between physical simulation time and playback time.
4. Evaluate responses using $\Phi$, $\Psi$, $A_r$, and mean radius across seeds, including constant-input and prescribed-protocol controls. Assess whether changes align with musical events without assuming causation or successful synchronization in advance.
5. Store feature-extraction settings, timestamps, mappings, seeds, and simulation parameters alongside exported results. Keep recordings separate from code and document their provenance and usage rights.

### Simulation and engineering work

- Separate configuration, force models, integration, metrics, and visualization; add a CLI and explicit data/figure export.
- Add controlled checks for force signs, pair symmetry, metric edge cases, and temporal switching; study numerical stability and timestep convergence.
- Remove hardcoded plot annotations and support consistent experiment metadata.
- Improve neighbor searches to reduce dense pairwise computation where appropriate.
- Explore population, density, and parameter sweeps with ensemble statistics, then compare with suitable observations or data before making empirical claims.

These items are planned work, not implemented capabilities. No songs are distributed with the project.

## License

Released under the MIT License. See [LICENSE](LICENSE).
