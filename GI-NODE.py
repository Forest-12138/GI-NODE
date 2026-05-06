import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from MovingMNIST import MovingMNIST

# --- Configuration ---
H, W = 64, 64
T = 20
SPF = 0.05
M = max(1, int(H * W * SPF))
LATENT_DIM = 256  # Increased capacity for vector latent space
NOISE_LEVEL = 0.02
SCALE_FACTOR = H * W / 2.0
TRAIN_BATCH = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_span = torch.linspace(0, 1, T).to(device)

# --- Data Loading ---
print("1. Loading Moving MNIST...")


def load_moving_mnist(train=True, batch_size=100):
    dataset = MovingMNIST(root='./data', train=train, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_tuple = next(iter(loader))
    full_seq = torch.cat((data_tuple[0], data_tuple[1]), dim=1)
    return full_seq.float().to(device) / 255.0


x_train_gt = load_moving_mnist(train=True, batch_size=TRAIN_BATCH)
x_test_gt = load_moving_mnist(train=False, batch_size=1)[0]
print(f"   Train: {x_train_gt.shape}, Test: {x_test_gt.shape}")

# --- Forward Measurement Model ---
print(f"2. Forward Model: SPF={SPF * 100}%, M={M}")
A = torch.randint(0, 2, (T, M, H, W)).float().to(device)
y_clean = torch.einsum('tmhw,thw->tm', A, x_test_gt) / SCALE_FACTOR
noise_std = NOISE_LEVEL * y_clean.std()
y_measured = y_clean + torch.randn_like(y_clean) * noise_std


# --- Network Architecture ---
# Using a flattened vector latent space to ensure stable prior constraints
# and clearer ODE integration compared to convolutional latent maps.

class Encoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(4, 64), nn.GELU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(4, 128), nn.GELU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(4, 256), nn.GELU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.GELU(),
            nn.Linear(512, LATENT_DIM)
        )

    def forward(self, x):
        s = x.shape[:-2]
        return self.net(x.reshape(-1, 1, H, W)).reshape(*s, LATENT_DIM)


class Decoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.GELU(),
            nn.Linear(512, 256 * 4 * 4), nn.GELU()
        )
        # Resize-Conv strategy to mitigate checkerboard artifacts
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1), nn.GroupNorm(4, 128), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1), nn.GroupNorm(4, 64), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, z):
        s = z.shape[:-1]
        x = self.fc(z.reshape(-1, LATENT_DIM)).reshape(-1, 256, 4, 4)
        return self.net(x).squeeze(1).reshape(*s, H, W)


class LatentODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        h = 512
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, LATENT_DIM)
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, z):
        return self.net(z) * self.scale


encoder = Encoder2D().to(device)
decoder = Decoder2D().to(device)
ode_func = LatentODEFunc().to(device)


def odeint_rk4(func, z0, t):
    """Explicit RK4 solver for consistent gradients during training and inference."""
    zs, z = [z0], z0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        k1 = func(t[i], z)
        k2 = func(t[i] + dt / 2, z + dt / 2 * k1)
        k3 = func(t[i] + dt / 2, z + dt / 2 * k2)
        k4 = func(t[i + 1], z + dt * k3)
        z = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        zs.append(z)
    return torch.stack(zs, dim=0)


# --- Two-Stage Training ---
print("\n4. Offline Training...")

# Phase A: Autoencoder Pre-training
print("   Stage A: AE pre-training...")
opt_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-5)
sched_ae = optim.lr_scheduler.CosineAnnealingLR(opt_ae, T_max=1000, eta_min=1e-5)

for epoch in range(1000):
    opt_ae.zero_grad()
    idx = torch.randperm(TRAIN_BATCH, device=device)[:64]
    x_mb = x_train_gt[idx]
    z = encoder(x_mb)
    xr = decoder(z)
    loss = nn.MSELoss()(xr, x_mb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
    opt_ae.step()
    sched_ae.step()

    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            loss_full = nn.MSELoss()(decoder(encoder(x_train_gt)), x_train_gt).item()
        print(f"   [AE] {epoch + 1}/1000 | loss={loss_full:.4f}")
        if loss_full < 0.006: break

# Phase B: Latent Trajectory Fitting (Freeze AE)
print("   Stage B: ODE trajectory fitting...")
encoder.eval()
for p in encoder.parameters(): p.requires_grad = False

with torch.no_grad():
    z_fixed = encoder(x_train_gt)
    dz_diag = z_fixed[:, 1:, :] - z_fixed[:, :-1, :]
    print(f"   Latent inter-frame dist: mean={dz_diag.norm(dim=-1).mean():.4f}")

opt_ode = optim.Adam(ode_func.parameters(), lr=5e-4)
sched_ode = optim.lr_scheduler.CosineAnnealingLR(opt_ode, T_max=1000, eta_min=1e-5)
best_ode_loss, best_ode_state = float('inf'), None

for epoch in range(1000):
    opt_ode.zero_grad()
    idx = torch.randperm(TRAIN_BATCH, device=device)[:32]
    z0 = z_fixed[idx, 0, :]
    z_gt = z_fixed[idx]
    z_pred = odeint_rk4(ode_func, z0, t_span).transpose(0, 1)

    loss = nn.MSELoss()(z_pred, z_gt)
    loss += 1e-4 * sum(p.norm() ** 2 for p in ode_func.parameters())  # Regularization

    loss.backward()
    torch.nn.utils.clip_grad_norm_(ode_func.parameters(), 0.5)
    opt_ode.step()
    sched_ode.step()

    if loss.item() < best_ode_loss:
        best_ode_loss = loss.item()
        best_ode_state = {k: v.clone() for k, v in ode_func.state_dict().items()}

ode_func.load_state_dict(best_ode_state)

# --- Online Iterative Reconstruction ---
print("\n5. Online Reconstruction...")

for model in [encoder, decoder, ode_func]:
    model.eval()
    for p in model.parameters(): p.requires_grad = False

# Compute prior statistics from training set
with torch.no_grad():
    z_all = encoder(x_train_gt)
    z0_all = z_all[:, 0, :]
    z0_mean = z0_all.mean(0)
    z0_std = z0_all.std(0) + 1e-6

# Global search for optimal z0 initialization
print("   Searching best z0...")
with torch.no_grad():
    records = []
    for i in range(TRAIN_BATCH):
        Z_c = odeint_rk4(ode_func, z0_all[i], t_span)
        x_c = decoder(Z_c)
        y_c = torch.einsum('tmhw,thw->tm', A, x_c) / SCALE_FACTOR
        l_meas = nn.MSELoss()(y_c, y_measured).item()
        bright = x_c.mean().item()
        records.append((l_meas, bright, i))

    valid = [r for r in records if r[1] > 0.03] or records
    valid.sort(key=lambda x: x[0])
    z0_init = z0_all[valid[0][2]].clone()

# Optimization Phase 1: Coarse Search
print("   Phase 1: Coarse search...")
z0_opt = nn.Parameter(z0_init.clone())
opt1 = optim.Adam([z0_opt], lr=5e-3)
best_z0, best_loss = z0_init.clone(), valid[0][0]

for epoch in range(400):
    opt1.zero_grad()
    Z_est = odeint_rk4(ode_func, z0_opt, t_span)
    x_est = decoder(Z_est)
    y_est = torch.einsum('tmhw,thw->tm', A, x_est) / SCALE_FACTOR

    loss_fid = nn.MSELoss()(y_est, y_measured)
    loss_prior = 0.05 * torch.mean(((z0_opt - z0_mean) / z0_std) ** 2)

    (loss_fid + loss_prior).backward()
    opt1.step()

    if loss_fid.item() < best_loss:
        best_loss, best_z0 = loss_fid.item(), z0_opt.detach().clone()

# Optimization Phase 2: Fine-tuning
print("   Phase 2: Fine-tuning...")
z0_opt = nn.Parameter(best_z0.clone())
opt2 = optim.Adam([z0_opt], lr=5e-4)

for epoch in range(300):
    opt2.zero_grad()
    y_est = torch.einsum('tmhw,thw->tm', A, decoder(odeint_rk4(ode_func, z0_opt, t_span))) / SCALE_FACTOR
    loss_fid = nn.MSELoss()(y_est, y_measured)
    loss_prior = 0.01 * torch.mean(((z0_opt - z0_mean) / z0_std) ** 2)
    (loss_fid + loss_prior).backward()
    opt2.step()
    if loss_fid.item() < best_loss:
        best_loss, best_z0 = loss_fid.item(), z0_opt.detach().clone()

# --- Visualization ---
with torch.no_grad():
    x_recon = decoder(odeint_rk4(ode_func, best_z0, t_span)).cpu().numpy()
x_gt = x_test_gt.cpu().numpy()

frames = [0, T // 4, T // 2, 3 * T // 4, T - 1]
fig, axes = plt.subplots(2, len(frames), figsize=(15, 6))
for i, f in enumerate(frames):
    axes[0, i].imshow(x_gt[f], cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    axes[1, i].imshow(x_recon[f], cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()