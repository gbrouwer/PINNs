# B_heat1d.py
import torch, math

import numpy as np, os
from plot_utils import plot_heatmap_grid, plot_timeseries
os.makedirs("figs", exist_ok=True)
torch.manual_seed(0)
device="cpu"

alpha = 0.1
Nx, Nt = 64, 64
x = torch.linspace(0,1,Nx)
t = torch.linspace(0,1,Nt)
X, T = torch.meshgrid(x, t, indexing='ij')
XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1).to(device)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 64), torch.nn.Tanh(),
    torch.nn.Linear(64, 64), torch.nn.Tanh(),
    torch.nn.Linear(64, 1)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def PDE_residual(x,t):
    x.requires_grad_(True); t.requires_grad_(True)
    u = model(torch.cat([x,t], dim=1))
    u_x  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_t  = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    return u_t - alpha*u_xx, u

bc_left  = torch.stack([torch.zeros(Nt), t], dim=1).to(device)
bc_right = torch.stack([torch.ones(Nt), t], dim=1).to(device)
ic       = torch.stack([x, torch.zeros(Nx)], dim=1).to(device)

for it in range(4000):
    opt.zero_grad()
    idx = torch.randint(0, XT.shape[0], (1024,))
    xt  = XT[idx]
    r_int, _ = PDE_residual(xt[:,0:1], xt[:,1:2])

    _, uL = PDE_residual(bc_left[:,0:1], bc_left[:,1:2])
    _, uR = PDE_residual(bc_right[:,0:1], bc_right[:,1:2])

    _, uI = PDE_residual(ic[:,0:1], ic[:,1:2])
    uI_true = torch.sin(torch.pi*ic[:,0:1])

    loss = (r_int**2).mean() + 50*((uL**2).mean() + (uR**2).mean()) + 50*((uI - uI_true)**2).mean()
    loss.backward(); opt.step()
    if (it+1)%1000==0:
        print(it+1, loss.item())
print("Done.")
# Visualization: sample dense grid, plot u(x,t) and residual heatmap
with torch.no_grad():
    xg = torch.linspace(0,1,128); tg = torch.linspace(0,1,128)
    Xg, Tg = torch.meshgrid(xg, tg, indexing='ij')
    XTg = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], dim=1)
    XTg = XTg.to(device)
    u = model(XTg).reshape(128,128).cpu().numpy()

# Compute residual heatmap
def residual_field(model, xg, tg):
    xg = xg.clone().requires_grad_(True).unsqueeze(1)
    tg = tg.clone().requires_grad_(True).unsqueeze(1)
    XT = torch.cartesian_prod(xg.squeeze(), tg.squeeze())
    X = XT[:,0:1]; Tt = XT[:,1:2]
    X.requires_grad_(True); Tt.requires_grad_(True)
    u = model(torch.cat([X,Tt], dim=1))
    u_x  = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0]
    u_t  = torch.autograd.grad(u, Tt, torch.ones_like(u), create_graph=True)[0]
    r = (u_t - alpha*u_xx).detach().cpu().numpy().reshape(len(xg), len(tg))
    return r

rmap = residual_field(model, torch.linspace(0,1,128), torch.linspace(0,1,128))

extent = (0,1,0,1)  # x in [0,1], t in [0,1]
plot_heatmap_grid(None, None, u, title="u(x,t)", fname="figs/B_u_heatmap.png", extent=extent)
plot_heatmap_grid(None, None, rmap, title="Residual heatmap", fname="figs/B_residual_heatmap.png", extent=extent)

# Time-slice at t=0.5
idx_t = 64
plot_timeseries(torch.linspace(0,1,128).numpy(), u[:,idx_t], ["u(x, t=0.5)"], "Mid-time profile", fname="figs/B_profile_tmid.png")
print("Saved figures in ./figs/: B_u_heatmap.png, B_residual_heatmap.png, B_profile_tmid.png")
