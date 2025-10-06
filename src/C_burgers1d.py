# C_burgers1d.py
import torch, math

import numpy as np, os
from plot_utils import plot_heatmap_grid, plot_timeseries
os.makedirs("figs", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1)

nu = 0.01 / math.pi
model = torch.nn.Sequential(
    torch.nn.Linear(2, 128), torch.nn.Tanh(),
    torch.nn.Linear(128,128), torch.nn.Tanh(),
    torch.nn.Linear(128,128), torch.nn.Tanh(),
    torch.nn.Linear(128,1)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=2e-3)

Nx, Nt = 256, 256
x = torch.linspace(-1,1,Nx, device=device)
t = torch.linspace(0,1,Nt, device=device)
X, T = torch.meshgrid(x, t, indexing='ij')
XT = torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)

def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def residual(x,t):
    x.requires_grad_(True); t.requires_grad_(True)
    u = model(torch.cat([x,t], dim=1))
    u_x  = grad(u, x); u_t = grad(u, t)
    u_xx = grad(u_x, x)
    r = u_t + u*u_x - nu*u_xx
    return r, u

ic = torch.stack([x, torch.zeros_like(x)], dim=1)
tb = t[:,None]
xb_left  = -torch.ones_like(tb)
xb_right =  torch.ones_like(tb)

for it in range(8000):
    opt.zero_grad()
    idx = torch.randint(0, XT.shape[0], (4096,), device=device)
    xt  = XT[idx]
    r_int, _ = residual(xt[:,0:1], xt[:,1:2])

    _, uI = residual(ic[:,0:1], ic[:,1:2])
    uI_true = -torch.sin(math.pi*ic[:,0:1])

    _, uL = residual(xb_left, tb)
    _, uR = residual(xb_right, tb)

    loss = (r_int**2).mean() + 50*((uL**2).mean() + (uR**2).mean()) + 50*((uI-uI_true)**2).mean()
    loss.backward(); opt.step()
    if (it+1)%1000==0: print(it+1, loss.item())
print("Done.")
# Visualization after training
with torch.no_grad():
    xg = torch.linspace(-1,1,256, device=device)
    tg = torch.linspace(0,1,256, device=device)
    Xg, Tg = torch.meshgrid(xg, tg, indexing='ij')
    XTg = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], dim=1)
    u = model(XTg).reshape(256,256).detach().cpu().numpy()

# Residual field
def residual_field(model, xg, tg):
    Xg, Tg = torch.meshgrid(xg, tg, indexing='ij')
    XT = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], dim=1).requires_grad_(True)
    u = model(XT)
    u_x = torch.autograd.grad(u, XT, torch.ones_like(u), create_graph=True, retain_graph=True)[0][:,0:1]
    u_t = torch.autograd.grad(u, XT, torch.ones_like(u), create_graph=True, retain_graph=True)[0][:,1:2]
    u_xx= torch.autograd.grad(u_x, XT, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:,0:1]
    r = (u_t + u*u_x - nu*u_xx).detach().cpu().numpy().reshape(len(xg), len(tg))
    return r

rmap = residual_field(model, torch.linspace(-1,1,256, device=device), torch.linspace(0,1,256, device=device))
extent = (-1,1,0,1)
plot_heatmap_grid(None, None, u, title="u(x,t) Burgers", fname="figs/C_u_heatmap.png", extent=extent)
plot_heatmap_grid(None, None, rmap, title="Residual heatmap", fname="figs/C_residual_heatmap.png", extent=extent)
plot_timeseries(np.linspace(-1,1,256), u[:,128], ["u(x, t=0.5)"], "Mid-time profile", fname="figs/C_profile_tmid.png")
print("Saved figures in ./figs/: C_u_heatmap.png, C_residual_heatmap.png, C_profile_tmid.png")
