# A_vehicle_inverse.py
# PINN for vehicle sensor consistency & inverse identification (PyTorch)
# - Synthetic generator for IMU (ax, ay, yaw rate), wheel speeds, steering, engine torque.
# - Physics: kinematic/dynamic bicycle model (flat road).
# - Goal: fit state trajectory and unknown parameters to data while minimizing physics residuals.
# - If you have real logs, replace `load_or_simulate_data()` to read your CSV.

import torch, math, sys
import os; os.makedirs("figs", exist_ok=True)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------- config ----------------------------
T = 12.0          # seconds
N = 800           # samples
dt = T/(N-1)
use_synthetic = True
noise_level = 0.02
# Parameter priors / bounds
param_init = dict(m=1500.0, Iz=2500.0, lf=1.2, lr=1.6, Cf=80000.0, Cr=80000.0, Cd=0.35, rho=1.2, Af=2.2)  # baseline
param_bounds = {
    "m": (800.0, 3000.0), "Iz": (1000.0, 6000.0),
    "lf": (0.8, 1.8), "lr": (1.0, 2.0),
    "Cf": (2e4, 2e5), "Cr": (2e4, 2e5),
    "Cd": (0.2, 0.6), "rho": (1.1, 1.3), "Af": (1.8, 3.0)
}

# ----------------------- synthetic data gen ---------------------
def load_or_simulate_data():
    t = torch.linspace(0, T, N, device=device).unsqueeze(1)
    # Inputs: steering delta (rad), engine wheel torque Tw (N*m), grade ~ 0
    delta = 0.08*torch.sin(0.6*2*math.pi*t) * (t>2)  # some steering
    Tw = 120.0 + 60.0*torch.sin(0.2*2*math.pi*t)     # gentle throttle variation
    # True parameters
    m, Iz, lf, lr = 1400.0, 2200.0, 1.2, 1.6
    Cf, Cr = 90000.0, 100000.0
    Cd, rho, Af = 0.32, 1.225, 2.1
    Rw = 0.3  # wheel radius
    # States: x, y, psi (yaw), vx, vy, r (yaw rate)
    X = torch.zeros(N,6, device=device)
    for i in range(1,N):
        x,y,psi,vx,vy,r = X[i-1]
        # tire slip angles (small-angle approx)
        beta_f = (vy + lf*r)/(vx+1e-3) - delta[i-1]
        beta_r = (vy - lr*r)/(vx+1e-3)
        Fyf = -Cf*beta_f
        Fyr = -Cr*beta_r
        # longitudinal forces
        Fx = (Tw[i-1]/Rw) - 0.5*rho*Cd*Af*vx*vx  # aero drag
        # dynamics
        ax = (Fx + Fyf*torch.sin(delta[i-1]))/m + vy*r
        ay = (Fyr + Fyf*torch.cos(delta[i-1]))/m - vx*r
        rdot = (lf*Fyf*torch.cos(delta[i-1]) + lf*Fyf*0.0 + (-lr)*Fyr)/Iz
        # integrate
        vx_n = torch.clamp(vx + ax*dt, 0.1, 80.0)
        vy_n = vy + ay*dt
        r_n  = r + rdot*dt
        psi_n= psi + r_n*dt
        x_n  = x + (vx_n*torch.cos(psi_n) - vy_n*torch.sin(psi_n))*dt
        y_n  = y + (vx_n*torch.sin(psi_n) + vy_n*torch.cos(psi_n))*dt
        X[i] = torch.stack([x_n,y_n,psi_n,vx_n,vy_n,r_n])
    # sensors: IMU ax, ay, yaw rate r; wheel speed (vx/Rw); steering delta; torque Tw
    ax_s = ((X[:,3][1:]-X[:,3][:-1])/dt).detach()
    ay_s = ((X[:,4][1:]-X[:,4][:-1])/dt + X[:-1,3]*X[:-1,5]).detach()
    r_s  = X[:,5].detach()
    w_s  = (X[:,3]/Rw).detach()
    # add noise
    def n(z): return z + noise_level*torch.randn_like(z)
    sensors = {
        "t": t.squeeze(),
        "ax": n(torch.cat([ax_s[:1], ax_s])),  # pad
        "ay": n(torch.cat([ay_s[:1], ay_s])),
        "r":  n(r_s),
        "omega": n(w_s),
        "delta": delta.squeeze()+0.0*torch.randn_like(t.squeeze())*0.0,
        "Tw": Tw.squeeze()+0.0*torch.randn_like(t.squeeze())*0.0
    }
    return sensors

data = load_or_simulate_data()

# ----------------------- PINN parameterization ------------------
# States as NN of time; outputs: x,y,psi,vx,vy,r
state_net = torch.nn.Sequential(
    torch.nn.Linear(1,128), torch.nn.Tanh(),
    torch.nn.Linear(128,128), torch.nn.Tanh(),
    torch.nn.Linear(128,128), torch.nn.Tanh(),
    torch.nn.Linear(128,6)
).to(device)

# Trainable physical parameters with soft-box constraints via sigmoid
def make_param(val, lo, hi):
    p = torch.nn.Parameter(torch.tensor(0.5))
    p._lo = lo; p._hi = hi; p._init = val
    with torch.no_grad():
        p.copy_(torch.tensor((val-lo)/(hi-lo)))
    return p

params = {k: make_param(param_init[k], *param_bounds[k]) for k in param_init.keys()}
Rw = 0.3  # fixed wheel radius

opt = torch.optim.Adam(list(state_net.parameters()) + list(params.values()), lr=1e-3)

# ---- Visualization helpers ----
import numpy as np
from plot_utils import plot_timeseries, plot_param_convergence

loss_hist = []
param_hist = {k: [] for k in params.keys()}


def squish(p):
    # map [0,1] via sigmoid to (lo,hi)
    s = torch.sigmoid(p)
    return p._lo + (p._hi - p._lo)*s

def time_grad(y, t):
    return torch.autograd.grad(y, t, torch.ones_like(y), create_graph=True)[0]

t = data["t"].to(device).unsqueeze(1)

for it in range(6000):
    opt.zero_grad()
    t.requires_grad_(True)
    x,y,psi,vx,vy,r = state_net(t).split(1, dim=1)

    # decode params
    m = squish(params["m"]); Iz = squish(params["Iz"])
    lf = squish(params["lf"]); lr = squish(params["lr"])
    Cf = squish(params["Cf"]); Cr = squish(params["Cr"])
    Cd = squish(params["Cd"]); rho = squish(params["rho"]); Af = squish(params["Af"])

    delta = data["delta"].to(device).unsqueeze(1)
    Tw    = data["Tw"].to(device).unsqueeze(1)

    # physics residuals
    beta_f = (vy + lf*r)/(vx+1e-3) - delta
    beta_r = (vy - lr*r)/(vx+1e-3)
    Fyf = -Cf*beta_f
    Fyr = -Cr*beta_r
    Fx = (Tw/Rw) - 0.5*rho*Cd*Af*vx*vx

    vx_t = time_grad(vx, t)
    vy_t = time_grad(vy, t)
    r_t  = time_grad(r, t)

    ax_phys = (Fx + Fyf*torch.sin(delta))/m + vy*r
    ay_phys = (Fyr + Fyf*torch.cos(delta))/m - vx*r
    rdot    = (lf*Fyf*torch.cos(delta) + (-lr)*Fyr)/Iz

    # data residuals (IMU & wheel speed)
    ax_err = ax_phys - data["ax"].to(device).unsqueeze(1)
    ay_err = ay_phys - data["ay"].to(device).unsqueeze(1)
    r_err  = r - data["r"].to(device).unsqueeze(1)
    w_err  = (vx/Rw) - data["omega"].to(device).unsqueeze(1)

    # dynamics consistency (tie time-derivatives to physics accelerations)
    dyn_err_vx = vx_t - ax_phys
    dyn_err_vy = vy_t - ay_phys
    dyn_err_r  = r_t  - rdot

    # regularize states
    reg = 1e-6*(x.pow(2)+y.pow(2)+psi.pow(2)+vx.pow(2)+vy.pow(2)+r.pow(2)).mean()

    loss = (ax_err.pow(2).mean()*50.0 +
            ay_err.pow(2).mean()*50.0 +
            r_err.pow(2).mean()*5.0  +
            w_err.pow(2).mean()*5.0  +
            dyn_err_vx.pow(2).mean()*1.0 +
            dyn_err_vy.pow(2).mean()*1.0 +
            dyn_err_r.pow(2).mean()*1.0 +
            reg)

    loss.backward(); opt.step()

    loss_val = loss.item()
    loss_hist.append(loss_val)
    for _k,_p in params.items():
        param_hist[_k].append(squish(_p).item())

    if (it+1)%1000==0:
        vals = {k: squish(v).item() for k,v in params.items()}
        print(f"Iter {it+1:5d} | loss {loss.item():.4e} | params {{"
              + ", ".join(f"{k}:{vals[k]:.3f}" for k in vals) + "}}")

print("Done. If loss is high or params unstable, your signals/model may be inconsistent.")

# ---- Post-training diagnostics ----
t_np = t.detach().cpu().numpy().reshape(-1)
ax_np = data["ax"].cpu().numpy().reshape(-1)
ay_np = data["ay"].cpu().numpy().reshape(-1)
r_np  = data["r"].cpu().numpy().reshape(-1)

with torch.no_grad():
    t.requires_grad_(False)
    x,y,psi,vx,vy,r = state_net(t).split(1, dim=1)
    # recompute physics accelerations
    m = squish(params["m"]); Iz = squish(params["Iz"])
    lf = squish(params["lf"]); lr = squish(params["lr"])
    Cf = squish(params["Cf"]); Cr = squish(params["Cr"])
    Cd = squish(params["Cd"]); rho = squish(params["rho"]); Af = squish(params["Af"])
    delta = data["delta"].to(device).unsqueeze(1)
    Tw    = data["Tw"].to(device).unsqueeze(1)
    beta_f = (vy + lf*r)/(vx+1e-3) - delta
    beta_r = (vy - lr*r)/(vx+1e-3)
    Fyf = -Cf*beta_f
    Fyr = -Cr*beta_r
    Fx = (Tw/0.3) - 0.5*rho*Cd*Af*vx*vx
    ax_phys = (Fx + Fyf*torch.sin(delta))/m + vy*r
    ay_phys = (Fyr + Fyf*torch.cos(delta))/m - vx*r
    ax_res = (ax_phys - data["ax"].to(device).unsqueeze(1)).cpu().numpy().reshape(-1)
    ay_res = (ay_phys - data["ay"].to(device).unsqueeze(1)).cpu().numpy().reshape(-1)
    vx_np = vx.cpu().numpy().reshape(-1)
    vy_np = vy.cpu().numpy().reshape(-1)
    r_est = r.cpu().numpy().reshape(-1)

# Save plots
plot_timeseries(t_np, [vx_np], ["vx"], "Estimated vx(t)", fname=r"figs/A_vx.png")
plot_timeseries(t_np, [vy_np], ["vy"], "Estimated vy(t)", fname=r"figs/A_vy.png")
plot_timeseries(t_np, [r_np, r_est], ["r_meas","r_est"], "Yaw rate: measured vs est.", fname=r"figs/A_r_compare.png")
plot_timeseries(t_np, [ax_res], ["ax residual"], "ax residual vs time", fname=r"figs/A_ax_residual.png")
plot_timeseries(t_np, [ay_res], ["ay residual"], "ay residual vs time", fname=r"figs/A_ay_residual.png")
plot_param_convergence(param_hist, "Parameter convergence", fname=r"figs/A_param_convergence.png")

print("Saved figures in ./figs/: A_vx.png, A_vy.png, A_r_compare.png, A_ax_residual.png, A_ay_residual.png, A_param_convergence.png")
