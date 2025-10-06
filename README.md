
# Physics-Informed Neural Networks (PINNs): A Practical, Intuitive, and Math-Grounded Guide

*Audience:* Intermediate → advanced ML practitioners who prefer geometric intuition but also want the core math. Emphasis on PyTorch and applications to sensor-consistency checks for autonomous systems, agent-based modeling (ABM) coarse-graining, and digital twins for flooding/flows.

---

## 1) Introduction

Physics-Informed Neural Networks (PINNs) augment standard neural-network training with *physics residuals* derived from governing equations—typically ODEs/PDEs plus auxiliary conditions such as initial and boundary constraints. Instead of relying solely on data loss, PINNs penalize violations of the physical laws, effectively restricting the learned function to a **manifold of physically admissible solutions**. This can yield strong data efficiency, improved extrapolation, and physically meaningful reconstructions—particularly when sensor data are sparse, noisy, or partially missing (Raissi, Perdikaris & Karniadakis, 2019; Karniadakis et al., 2021).

Geometrically, if the unconstrained hypothesis space is a vast landscape, then adding physics defines a lower-dimensional surface within it; optimization proceeds *on or near* that surface. In practice, PINNs use automatic differentiation to compute derivatives of the network outputs with respect to their inputs and insert those derivatives into PDE residuals.

**Canonical PINN loss**
\[
\mathcal{L}(\theta,\lambda) \;=\; \lambda_{\text{data}} \,\Vert u_\theta - u^{\text{obs}}\Vert^2
\;+\; \lambda_{\text{PDE}} \,\Vert \mathcal{F}[u_\theta;\lambda] \Vert^2
\;+\; \lambda_{\text{BC/IC}} \,\Vert \mathcal{B}[u_\theta]\Vert^2
\;+\; \lambda_{\text{reg}} \,\Vert\theta\Vert^2,
\]
where \(u_\theta\) is the neural approximation (e.g., \(u:\mathbb{R}^d\!\to\!\mathbb{R}^m\)), \(\mathcal{F}\) is the differential operator, \(\lambda\) denotes physical parameters (possibly unknown), and \(\mathcal{B}\) encodes BC/IC constraints (Raissi, Perdikaris & Karniadakis, 2019).

---

## 2) Why use Physics-Informed Neural Networks?

- **Data efficiency and inductive bias.** Physical laws act as powerful priors that curb overfitting and help generalization, especially OOD (Karniadakis et al., 2021).
- **Inverse problems and system identification.** PINNs can estimate hidden states, parameters, missing boundary conditions, and even unknown source terms by jointly minimizing data mismatch and physics residuals (Raissi, Perdikaris & Karniadakis, 2019).
- **Sensor validation and reconstruction.** When asked, “Do these measurements admit any physically plausible trajectory/field?”, PINNs provide a principled answer: large, persistent residuals or non-identifiable parameters indicate inconsistency with the physics; a stable low-residual fit indicates feasibility (Falas et al., 2023; Iliadis et al., 2025).
- **End-to-end differentiability.** AD supplies exact derivatives for residuals and sensitivities without hand-coded finite differences.
- **Hybridization with operator learning.** PINNs pair naturally with Neural Operators (FNO/DeepONet) to create fast, physics-consistent surrogates across parameterized families of PDEs (Li et al., 2020; Lu et al., 2021; Li et al., 2021).

---

## 3) History and Background

The PINN framework was formalized by Raissi, Perdikaris & Karniadakis (2019) following earlier preprints (2017). The field expanded rapidly under the umbrella of *physics-informed machine learning* (PIML) with comprehensive reviews by Karniadakis et al. (2021). Tooling matured via libraries such as **DeepXDE** (Lu et al., 2021). Subsequent work analyzed failure modes and optimization pathologies—e.g., gradient imbalance, stiffness, and multi-scale difficulties—and proposed mitigation strategies such as curriculum training and adaptive loss balancing (Wang, Teng & Perdikaris, 2021; Krishnapriyan et al., 2021).

In parallel, **Neural Operators** (e.g., FNO) and **DeepONet** generalized learning to *operators* that map functions to functions across geometries and coefficients, with **PINO** hybridizing operator learning and physics residuals (Li et al., 2020; Lu et al., 2021; Li et al., 2021).

---

## 4) Comparison with Other Modern Neural Network Paradigms

- **Plain supervised nets.** Require many labels and typically lack physical consistency; PINNs enforce governing laws and constraints (Karniadakis et al., 2021).
- **Sparse model discovery (e.g., SINDy).** Learns symbolic dynamics from data with interpretable terms; struggles in high-noise/high-dimensional regimes. Complementary to PINNs: SINDy proposes candidate terms; PINNs enforce the resulting physics during learning (Brunton, Proctor & Kutz, 2016).
- **Neural Operators (FNO/DeepONet).** Learn solution operators over families of PDEs and mesh resolutions; excellent for fast surrogates. Augment with physics residuals (PINO) for better physical fidelity and generalization (Li et al., 2020; Lu et al., 2021; Li et al., 2021).
- **GNNs on physical graphs.** Encode conservation via message passing; useful for networks (power/traffic). PINNs target differential constraints in continuous domains; hybrids exist.

---

## 5) Overview of Methodologies

### 5.1 Soft vs. hard constraints
- **Soft constraints** insert IC/BC and PDE residuals as penalties in the loss (standard PINN setup).
- **Hard constraints** impose BCs analytically via change-of-variables or architecture design, reducing loss-competition and improving convergence when feasible (Raissi, Perdikaris & Karniadakis, 2019).

### 5.2 Collocation design
Interior points (for PDE residuals) and boundary/initial points (for constraints) are sampled—often via Latin hypercube or stratified schemes. Adaptive sampling can focus on high-residual regions.

### 5.3 Inverse PINNs
Treat physical parameters or unknown forcings as trainable variables (or via auxiliary subnetworks). The loss guides both fields and parameters to fit data and physics simultaneously (Raissi, Perdikaris & Karniadakis, 2019).

### 5.4 Multi-fidelity and hybrid training
Combine sparse high-quality measurements with abundant low-fidelity simulations; or couple coarse supervised operator learning with fine-resolution physics residuals (PINO) (Li et al., 2021).

### 5.5 Optimization tactics
- Optimizers: ADAM followed by (L-)BFGS is common.
- Loss-balancing/annealing and curriculum (from easier physics to harder) mitigate gradient pathologies (Wang, Teng & Perdikaris, 2021; Krishnapriyan et al., 2021).
- Architectural cues: smooth activations (tanh/sine), Fourier features for high-frequency content, residual/skip connections.

---

## 6) Use Cases (with your focus)

- **Sensor validation and event reconstruction for vehicles/drones.** Given IMU/odometry/actuator logs, train an inverse PINN that recovers latent trajectories and parameters under rigid-body + tire/airframe dynamics. If no low-residual fit exists or estimates become unstable, the data/model are inconsistent—flagging sensor faults or time-base errors (Falas et al., 2023; Iliadis et al., 2025).
- **State estimation / observers.** Physics-informed observers combine partial measurements with dynamics to infer hidden states (e.g., vehicle sideslip, battery SOC/SOH, power-system states).
- **Digital twins for flooding.** Enforce shallow-water equations (SWE) to reconstruct water depth/velocity from sparse gauges and rainfall/terrain, enabling nowcasting and what-if analysis (Qi et al., 2024; Tian et al., 2025; Li et al., 2024).
- **ABM coarse-graining.** Learn PDE-level conservation equations that summarize agent flows; enforce them via PINNs to fuse sparse telemetry with macro-scale physics.
- **Physical logistics & air traffic.** While separation rules are algorithmic, macroscopic traffic densities often obey conservation-like PDEs; PINNs can regularize such macro models with real counts/queues.

---

## 7) Future Headings (Future Directions)

- **Scalability and stiffness.** Domain decomposition, adaptive sampling, and curriculum schedules to address multi-scale PDEs and sharp features (Wang, Teng & Perdikaris, 2021).
- **Operator-PINN hybrids.** PINO and other physics-regularized operators for rapid surrogates with embedded constraints across domains and meshes (Li et al., 2021).
- **Uncertainty quantification.** Bayesian PINNs and ensembles for decision-critical autonomy and digital twins.
- **Discrete/inequality physics.** Contact, saturations, complementarity—beyond smooth PDEs.
- **Theory and benchmarks.** Approximation/generalization guarantees and standardized suites akin to those for neural operators.

---

## 8) Core Theory and Math

### 8.1 Network structure and architecture
Most PINNs are smooth MLPs \(u_\theta:\mathbb{R}^d\!\to\!\mathbb{R}^m\) with tanh/sine activations to stabilize higher-order derivatives. Fourier features, residual connections, and positional encodings help represent multi-scale structure. For families of PDEs or varying geometries, **Neural Operators** (FNO/DeepONet) are attractive; physics-regularized variants (PINO) combine both worlds (Li et al., 2020; Lu et al., 2021; Li et al., 2021).

### 8.2 Typical inputs and outputs
- **Inputs:** coordinates \((x,t)\), geometry/coefficients, sensor timestamps, known forcings \(f(x,t)\).
- **Outputs:** state fields (e.g., \(u,v,p,h\)), latent states, or parameters \( \lambda\).

### 8.3 Training of PINNs
Given
\[
\mathcal{F}\big(u,\nabla u,\nabla^2 u;\lambda\big)=0,
\quad \mathcal{B}(u)=0,
\]
define residuals \(r(x,t;\theta,\lambda)=\mathcal{F}(u_\theta,\nabla u_\theta,\nabla^2 u_\theta;\lambda)\).
The composite loss (Intro) is minimized over interior collocation points (for \(r\)), boundary/initial points (for \(\mathcal{B}\)), and data points (for fit). AD computes all needed derivatives. When enforcing hard BCs, reparameterize \(u_\theta\) to satisfy \(\mathcal{B}\equiv 0\) identically (Raissi, Perdikaris & Karniadakis, 2019).

### 8.4 Testing accuracy / predictive power
- **Residual norms** on unseen collocation sets.
- **Task metrics** (RMSE on held-out sensors/gauges).
- **Constraint satisfaction** (IC/BC error; energy/mass balance).
- **Ablations** (with/without physics; with/without certain sensors).
- **Posterior checks** if Bayesian/ensembled.

### 8.5 Pitfalls and cautions
- **Gradient pathologies and competing losses.** Composite losses can be ill-conditioned; adopt adaptive weighting, curriculum, and domain decomposition (Wang, Teng & Perdikaris, 2021; Krishnapriyan et al., 2021).
- **Multi-scale features and sharp fronts.** Vanilla MLPs struggle; add positional/Fourier features or switch to operator models; consider hybrid PINO training (Li et al., 2020; Li et al., 2021).
- **Identifiability in inverse problems.** Several parameters may be confounded, especially under weak excitation; add priors, bounds, regularization, or controlled maneuvers to improve observability.

---

## 9) Four Worked-Out Toy Examples (PyTorch overview)

> Full runnable scripts are provided separately; see:
> - **A_vehicle_inverse.py** — Vehicle sensor-consistency inverse PINN (CPU-capable).  
> - **B_heat1d.py** — 1D heat equation PINN (CPU).  
> - **C_burgers1d.py** — 1D viscous Burgers’ (GPU recommended).  
> - **D_shallow_water_2d.py** — 2D shallow-water (GPU strongly recommended).  

### 9.1 Example A (sensor consistency, inverse identification)
A kinematic/dynamic bicycle-model PINN jointly fits state trajectories and parameters from IMU/wheel-speed/steering/torque logs. **Interpretation:** If residuals remain large or parameters drift non-physically, the data or model are inconsistent (e.g., sensor fault, timestamp drift).

### 9.2 Example B (forward PDE: 1D heat)
A minimal PINN that enforces \(u_t=\alpha u_{xx}\) with Dirichlet BCs and a sinusoidal IC using only physics and boundary/initial penalties.

### 9.3 Example C (stiffer PDE: Burgers’)
Illustrates optimization challenges from nonlinearity and small viscosity; demonstrates adaptive weighting/curriculum.

### 9.4 Example D (digital twin: 2D shallow-water)
Outputs water depth/velocity given DEM and rainfall; add gauge data terms for assimilation and Manning friction for realism. Suitable for flood nowcasting and what-if analyses.

---

## Practical guidance for your projects

1. **Vehicle/drone data sanity checks.** Pose the problem as an inverse PINN with your dynamics and actuator limits; add bias/drift parameters for sensors; examine residual heatmaps over time for fault localization (Falas et al., 2023; Iliadis et al., 2025).
2. **Flood twins.** Start with coarse grids and synthetic rainfall; introduce real DEM and gauge data; add Manning friction and no-flux/open BCs; consider PINO for broader scenario coverage (Qi et al., 2024; Li et al., 2021).
3. **ABM mesoscales.** Fit a PDE surrogate to agent-density fields; regularize with conservation residuals; use operator learning for fast rollout across policies.

---

## References (inline citations)

- Raissi, Perdikaris & Karniadakis (2019) — PINN foundations.  
- Karniadakis et al. (2021) — Physics-informed ML review/outlook.  
- Lu et al. (2021) — DeepXDE library.  
- Wang, Teng & Perdikaris (2021); Krishnapriyan et al. (2021) — Gradient pathologies/failure modes and mitigations.  
- Li et al. (2020) — Fourier Neural Operator (FNO).  
- Lu et al. (2021) — DeepONet operator learning.  
- Li et al. (2021) — PINO hybrid operator learning + physics.  
- Qi et al. (2024); Tian et al. (2025) — PINNs for shallow-water/flood modeling.  
- Brunton, Proctor & Kutz (2016) — SINDy for sparse model discovery.  
- Falas et al. (2023); Iliadis et al. (2025) — Physics-informed state estimation/observers in power/controls (as exemplars for sensor/observer roles).

---

## Bibliography

- **Brunton, S.L., Proctor, J.L., & Kutz, J.N. (2016).** Discovering governing equations from data by sparse identification of nonlinear dynamical systems (SINDy). *PNAS*, **113**(15): 3932–3937.

- **Falas, S., et al. (2023).** Physics-Informed Neural Networks for Accelerating Power System State Estimation. *arXiv:2310.03088*.

- **Iliadis, P., et al. (2025).** PINNs for enhanced state estimation in distribution systems. *Applied Sciences*, **15**(13): 7507.

- **Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021).** Physics-informed machine learning. *Nature Reviews Physics*, **3**: 422–440.

- **Krishnapriyan, A.S., Gholami, A., Zhe, S., Kirby, R.M., & Mahoney, M.W. (2021).** Characterizing possible failure modes in PINNs. *NeurIPS 34*. 

- **Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020).** Fourier Neural Operator for parametric PDEs. *arXiv:2010.08895* (JMLR 2023 extensions).

- **Li, Z., Zheng, H., Kovachki, N., Jin, D., Chen, H., Liu, B., Azizzadenesheli, K., & Anandkumar, A. (2021).** Physics-Informed Neural Operator for learning PDEs. *arXiv:2111.03794*.

- **Lu, L., Meng, X., Mao, Z., & Karniadakis, G.E. (2021).** DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, **63**: 208–228.

- **Lu, L., Jin, P., & Karniadakis, G.E. (2021).** Learning nonlinear operators via DeepONet. *Nature Machine Intelligence*, **3**: 218–229.

- **Qi, X., et al. (2024).** Physics-informed neural networks for free-surface flows (2D SWE). *Journal of Hydrology*. 

- **Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019).** Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. *Journal of Computational Physics*, **378**: 686–707.

- **Tian, Y., et al. (2025).** Physics-Informed Neural Networks for Solving the Two-Dimensional Shallow Water Equations. *arXiv:2501.11372*.

- **Wang, S., Teng, Y., & Perdikaris, P. (2021).** Understanding and mitigating gradient flow pathologies in PINNs. *SIAM Journal on Scientific Computing*, **43**(5): A3055–A3081.

---

### Appendix: Minimal PINN training loop (pseudocode)

```python
for iter in range(max_iters):
    # sample interior and boundary/initial batches
    x_int, t_int = sample_interior(batch_int)
    x_b, t_b, g_b = sample_boundary(batch_bc)

    # forward
    u = model(cat(x_int, t_int))
    r = PDE_residual(u, x_int, t_int)         # via auto-diff
    u_b = model(cat(x_b, t_b))

    # loss
    L = w_pde * mse(r, 0) + w_bc * mse(u_b, g_b) + w_data * mse(u_data, y_data)

    # step
    L.backward(); opt.step(); opt.zero_grad()
```

For runnable code, see the four example scripts shipped alongside this chapter.
