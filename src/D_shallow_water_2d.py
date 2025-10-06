# D_shallow_water_2d.py
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

g = 9.81

class Net(torch.nn.Module):
    def __init__(self, in_dim=3, out_dim=3, width=256, depth=6):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, width), torch.nn.Tanh()]
        for _ in range(depth-2):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, out_dim)]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, xyt): return self.net(xyt)

model = Net().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

Nx, Ny, Nt = 64, 64, 32
x = torch.linspace(0,1,Nx, device=device)
y = torch.linspace(0,1,Ny, device=device)
t = torch.linspace(0,1,Nt, device=device)

X, Y = torch.meshgrid(x, y, indexing='ij')
zb = 0.05*X  # gentle slope

def rainfall(xyt):
    return 1.0*torch.exp(-50*((xyt[:,0:1]-0.5)**2 + (xyt[:,1:2]-0.5)**2))*(xyt[:,2:3]>0.2)*(xyt[:,2:3]<0.6)

def grads(w, vars):
    return torch.autograd.grad(w, vars, torch.ones_like(w), create_graph=True)[0]

for it in range(6000):
    opt.zero_grad()

    bx = x[torch.randint(0,Nx,(2048,), device=device)]
    by = y[torch.randint(0,Ny,(2048,), device=device)]
    bt = t[torch.randint(0,Nt,(2048,), device=device)]
    XYT = torch.stack([bx,by,bt], dim=1).requires_grad_(True)
    out = model(XYT); h, u, v = out[:,0:1], out[:,1:2], out[:,2:3]

    hx = grads(h, XYT)[:,0:1]; hy = grads(h, XYT)[:,1:2]; ht = grads(h, XYT)[:,2:3]
    ux = grads(u, XYT)[:,0:1]; uy = grads(u, XYT)[:,1:2]; ut = grads(u, XYT)[:,2:3]
    vx = grads(v, XYT)[:,0:1]; vy = grads(v, XYT)[:,1:2]; vt = grads(v, XYT)[:,2:3]

    div_hu = (h*ux + u*hx) + (h*vy + v*hy)
    R = rainfall(XYT)

    zb_x = 0.05*torch.ones_like(h); zb_y = torch.zeros_like(h)

    r_cont = ht + div_hu - R
    r_mx   = (h*ut + u*ht) + (h*(u*ux + v*uy) + g*h*hx) + g*h*zb_x
    r_my   = (h*vt + v*ht) + (h*(u*vx + v*vy) + g*h*hy) + g*h*zb_y

    loss = (r_cont**2).mean() + 0.1*((r_mx**2).mean() + (r_my**2).mean())
    loss.backward(); opt.step()
    if (it+1)%1000==0: print(it+1, loss.item())

print("Done. Replace zb and rainfall() with real DEM/rain to build a flood-twin.")