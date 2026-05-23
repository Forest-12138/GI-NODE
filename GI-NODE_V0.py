import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision.datasets import MNIST
from torchvision import transforms

# ==========================================
# 0. 全局参数
# ==========================================
H, W = 64, 64
T = 20
SPF = 0.25
M = max(1, int(H * W * SPF))
LATENT_DIM = 16  # 论文 2D 流实验中 Latent Dim 设为 64
NOISE_LEVEL = 0.02
SCALE_FACTOR = H * W / 2.0
TRAIN_BATCH = 1024
MINI_BATCH = 16  # 联合训练显存占用大，MB 设为 16 以适配常规 GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_span = torch.linspace(0, 1, T).to(device)
## ==========================================
# 1. 数据加载（空心圆环的阻尼螺旋运动 ODE 轨迹）
# ==========================================
print("1. Generating Damped Spiral Hollow Ring Dataset...")

def generate_spiral_ring_dataset(num_samples, T, H, W):
    seqs = []
    # --- 阻尼螺旋的运动学参数 ---
    R_traj = 24.0            # 初始轨道半径设定大一点，从边缘向内螺旋
    omega = 2 * np.pi * 1.5  # 设定较高的角速度，保证在 T=1 内能转一圈半
    gamma = 1.5              # 阻尼系数，控制向内收缩的速率
    t = torch.linspace(0, 1, T)

    # --- 空心圆环的几何参数 ---
    R_ring = 6.0             # 圆环的中心半径
    thickness_sigma = 1.5    # 圆环边缘的高斯厚度

    for _ in range(num_samples):
        # 引入初始相位和轨道半径的随机性
        phase = torch.rand(1) * 2 * np.pi
        r_init = R_traj + (torch.randn(1) * 2.0)

        frames = []
        for t_step in t:
            # ODE 解析解：半径指数衰减，角度线性增加
            r_t = r_init * torch.exp(-gamma * t_step)
            angle_t = omega * t_step + phase

            # 转换为笛卡尔坐标
            cx = H / 2.0 + r_t * torch.cos(angle_t)
            cy = W / 2.0 + r_t * torch.sin(angle_t)

            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

            # 计算每个像素点到物体中心的欧氏距离，并生成空心圆环
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            ring = torch.exp(-(dist - R_ring) ** 2 / (2 * thickness_sigma ** 2))
            frames.append(ring)

        seqs.append(torch.stack(frames, dim=0))

    return torch.stack(seqs, dim=0).to(device)

x_train_gt = generate_spiral_ring_dataset(TRAIN_BATCH, T, H, W)  # [B, T, H, W]
x_test_gt = generate_spiral_ring_dataset(1, T, H, W)[0]          # [T, H, W]
print(f"   Train: {x_train_gt.shape}, Test: {x_test_gt.shape}")

# ==========================================
# 2. 前向模型
# ==========================================
print(f"2. Forward Model: SPF={SPF * 100}%, M={M}")
A = torch.randint(0, 2, (T, M, H, W)).float().to(device)
y_clean = torch.einsum('tmhw,thw->tm', A, x_test_gt) / SCALE_FACTOR
noise_std = NOISE_LEVEL * y_clean.std()
y_measured = y_clean + torch.randn_like(y_clean) * noise_std


class Encoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        # 加回 GroupNorm 稳定方差，防止零向量坍塌
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GroupNorm(4, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(4, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(4, 256), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.SiLU(),
            nn.Linear(512, LATENT_DIM)
        )

    def forward(self, x):
        s = x.shape[:-2]
        z = self.net(x.reshape(-1, 1, H, W))
        return z.reshape(*s, LATENT_DIM)


class Decoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM, 512), nn.SiLU(),
            nn.Linear(512, 256 * 4 * 4), nn.SiLU()
        )

        def res_block(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(4, c), nn.SiLU(),
                nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(4, c), nn.SiLU()
            )

        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.rb1 = res_block(128)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.rb2 = res_block(64)
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.rb3 = res_block(32)
        self.up4 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.rb4 = res_block(16)
        self.out = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, z):
        s = z.shape[:-1]
        x = self.fc(z.reshape(-1, LATENT_DIM)).reshape(-1, 256, 4, 4)
        x = self.up1(x);
        x = self.rb1(x) + x
        x = self.up2(x);
        x = self.rb2(x) + x
        x = self.up3(x);
        x = self.rb3(x) + x
        x = self.up4(x);
        x = self.rb4(x) + x
        return self.out(x).squeeze(1).reshape(*s, H, W)


class LatentODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        h = 256
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, h), nn.SiLU(),  # 论文2D动力学使用ReLU/SiLU
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, LATENT_DIM)
        )

    def forward(self, t, z):
        return self.net(z)


encoder = Encoder2D().to(device)
decoder = Decoder2D().to(device)
ode_func = LatentODEFunc().to(device)


def odeint_rk4(func, z0, t):
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


# ==========================================
# 4. 离线训练 (方案 A: 强制解耦与冻结解码器)
# ==========================================
print("\n4. Offline Training (Scheme A: Decoupled AE and ODE)...")

criterion_bce = nn.BCEWithLogitsLoss()
criterion_mse = nn.MSELoss()
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
_AMP = 'cuda' if torch.cuda.is_available() else 'cpu'
# ---------------------------------------------------------
# Phase 4.1: Static AE Pre-training (Learning Sharp Shapes)
# ---------------------------------------------------------
print("   [Phase 4.1] Static AE Pre-training (Strict BCE for Sharpness)...")
x_train_static = x_train_gt.reshape(-1, H, W)
STATIC_BATCH = 256  # 扩大 Batch Size 以稳定 BCE 梯度
STATIC_EPOCHS = 500  # 延长训练时间，确保流形被完全建立

opt_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-5)
sched_ae = optim.lr_scheduler.CosineAnnealingLR(opt_ae, T_max=STATIC_EPOCHS, eta_min=1e-5)

# 仅使用 BCE 损失，因为 MNIST 本质上是二值图像
criterion_bce = nn.BCEWithLogitsLoss()

for epoch in range(STATIC_EPOCHS):
    idx = torch.randperm(x_train_static.size(0), device=device)[:STATIC_BATCH]
    x_mb = x_train_static[idx]

    opt_ae.zero_grad()
    with torch.amp.autocast(device_type=_AMP, enabled=torch.cuda.is_available()):
        z = encoder(x_mb.unsqueeze(1)).squeeze(1)
        logits = decoder(z.unsqueeze(1)).squeeze(1)

        # 抛弃 MSE 和 L1，纯粹用 BCE 逼迫网络输出锐利的黑白二值边界
        loss = criterion_mse(logits, x_mb) + 1e-4 * torch.mean(z**2)

    scaler.scale(loss).backward()
    scaler.unscale_(opt_ae)
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
    scaler.step(opt_ae)
    scaler.update()
    sched_ae.step()

    if (epoch + 1) % 50 == 0:
        print(f"      AE Epoch {epoch + 1:3d}/{STATIC_EPOCHS} | Static AE BCE Loss = {loss.item():.5f}")

# ==========================================
# 调试检查点：强制可视化 AE 的静态重建能力
# ==========================================
# 在进入 Phase 4.2 之前，我们必须确保 Decoder 真的学会了锐利形状！
with torch.no_grad():
    test_static_img = x_train_static[:5]
    test_z = encoder(test_static_img.unsqueeze(1)).squeeze(1)
    test_recon = torch.clamp(decoder(test_z.unsqueeze(1)).squeeze(1), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(test_static_img[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title("GT")
        axes[0, i].axis('off')
        axes[1, i].imshow(test_recon[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title("AE Recon")
        axes[1, i].axis('off')
    plt.suptitle("Sanity Check: Static AE Reconstruction (MUST BE SHARP)")
    plt.tight_layout()
    plt.savefig("ae_sanity_check.png")
    plt.close()
print("   [Sanity Check] Saved 'ae_sanity_check.png'. CHECK THIS IMAGE before trusting the ODE.")
# ---------------------------------------------------------
# Phase 4.2: 动态 ODE 训练 (彻底冻结 AE 流形)
# ---------------------------------------------------------
print("\n   [Phase 4.2] Dynamic ODE Training (Frozen AE Manifold)...")

# 【核心修改】：彻底冻结 Encoder 和 Decoder，只训练 ODE！
encoder.eval()
decoder.eval()
for p in encoder.parameters():
    p.requires_grad = False
for p in decoder.parameters():
    p.requires_grad = False

# 优化器现在只负责更新 ODE 的参数，学习率稍微调高加快收敛
opt_dyn = optim.Adam(ode_func.parameters(), lr=1e-3, weight_decay=1e-5)
sched_dyn = optim.lr_scheduler.CosineAnnealingLR(opt_dyn, T_max=700, eta_min=1e-5)

DYN_EPOCHS = 700
for epoch in range(DYN_EPOCHS):
    idx = torch.randperm(TRAIN_BATCH, device=device)[:MINI_BATCH]
    x_mb = x_train_gt[idx]  # [MB, T, H, W]

    opt_dyn.zero_grad()
    with torch.amp.autocast(device_type=_AMP, enabled=torch.cuda.is_available()):

        # 1. 获取目标隐变量轨迹 (由于 Encoder 冻结，需使用 no_grad 节省显存)
        with torch.no_grad():
            z_seq = encoder(x_mb)

        # 2. ODE 积分预测
        z0 = z_seq[:, 0, :]
        z_pred = odeint_rk4(ode_func, z0, t_span).transpose(0, 1)

        # 3. 核心损失：让 ODE 预测的轨迹严格贴合真实的隐空间轨迹
        loss_latent = criterion_mse(z_pred, z_seq)

        # 4. 辅助损失：图像域的重建误差
        logits_pred = decoder(z_pred)
        loss_pred_img = criterion_mse(logits_pred, x_mb)

        # 赋予 Latent 误差更高的权重，强迫 ODE 学习动力学
        loss = loss_latent * 10.0 + loss_pred_img

    scaler.scale(loss).backward()
    scaler.unscale_(opt_dyn)
    torch.nn.utils.clip_grad_norm_(ode_func.parameters(), 1.0)
    scaler.step(opt_dyn)
    scaler.update()
    sched_dyn.step()

    if (epoch + 1) % 50 == 0:
        print(f"      DYN Epoch {epoch + 1:3d}/{DYN_EPOCHS} | Total = {loss.item():.5f} "
              f"(Img_Pred = {loss_pred_img.item():.5f}, Latent = {loss_latent.item():.5f})")

print("   Offline Training Complete.")

# 5. 在线重建 (Eq. 7) 改进版：智能初始化 + 两阶段优化
# ==========================================
print("\n5. Online Reconstruction (Eq. 7) with Smart Init...")
for model in [encoder, decoder, ode_func]:
    model.eval()
    for p in model.parameters(): p.requires_grad = False

# --- 步骤 5.1: 智能初始化 (在潜空间中寻找最近邻) ---
print("   Searching for best initialization candidate...")
best_init_loss = float('inf')
best_z_init = None

with torch.no_grad():
    # 扫描前 256 个训练样本，寻找一个具有良好“结构”的起点
    for i in range(0, 256, MINI_BATCH):
        x_cand = x_train_gt[i:i + MINI_BATCH]
        z_cand = encoder(x_cand)  # [MB, T, D]

        for j in range(z_cand.size(0)):
            x_est = torch.clamp(decoder(z_cand[j:j + 1]), 0, 1).squeeze(0)
            y_est = torch.einsum('tmhw,thw->tm', A, x_est) / SCALE_FACTOR
            l = nn.MSELoss()(y_est, y_measured).item()
            if l < best_init_loss:
                best_init_loss = l
                best_z_init = z_cand[j].clone()

print(f"   Found init candidate with Measurement MSE: {best_init_loss:.6f}")

# --- 步骤 5.2: Phase 1 - 仅优化 z0 (强制物理流形) ---
# 这一步阻止模型产生“模糊光斑”，迫使它在真实的数字流形上滑动
print("   Phase 1: Optimizing z0 (Strict ODE Manifold)...")
z0_opt = nn.Parameter(best_z_init[0].clone())
opt_z0 = optim.Adam([z0_opt], lr=1e-2)
sched_z0 = optim.lr_scheduler.CosineAnnealingLR(opt_z0, T_max=200, eta_min=1e-4)

for epoch in range(200):
    opt_z0.zero_grad()
    Z_ode = odeint_rk4(ode_func, z0_opt, t_span)  # [T, D]
    x_est = torch.clamp(decoder(Z_ode.unsqueeze(0)), 0, 1).squeeze(0)
    y_est = torch.einsum('tmhw,thw->tm', A, x_est) / SCALE_FACTOR
    loss_meas = nn.MSELoss()(y_est, y_measured)

    loss_meas.backward()
    opt_z0.step()
    sched_z0.step()

# --- 步骤 5.3: Phase 2 - 联合优化完整 Z 序列 (论文公式 7) ---
print("   Phase 2: Optimizing full Z sequence (Eq. 7)...")
with torch.no_grad():
    # 将 Phase 1 得到的完美物理轨迹作为 Phase 2 的起点
    Z_opt_init = odeint_rk4(ode_func, z0_opt.detach(), t_span)

Z_opt = nn.Parameter(Z_opt_init.clone())
opt_recon = optim.Adam([Z_opt], lr=5e-3)
sched_recon = optim.lr_scheduler.CosineAnnealingLR(opt_recon, T_max=300, eta_min=1e-5)

LAMBDA_ODE = 1.0
best_recon_loss = float('inf')
best_Z = Z_opt.detach().clone()

for epoch in range(300):
    opt_recon.zero_grad()

    # 测量保真度项
    x_est = torch.clamp(decoder(Z_opt.unsqueeze(0)), 0, 1).squeeze(0)
    y_est = torch.einsum('tmhw,thw->tm', A, x_est) / SCALE_FACTOR
    loss_meas = nn.MSELoss()(y_est, y_measured)

    # 潜变量动态正则化项
    Z_ode = odeint_rk4(ode_func, Z_opt[0], t_span)
    loss_ode = nn.MSELoss()(Z_opt, Z_ode)

    loss_total = loss_meas + LAMBDA_ODE * loss_ode

    loss_total.backward()
    torch.nn.utils.clip_grad_norm_([Z_opt], 0.5)
    opt_recon.step()
    sched_recon.step()

    if loss_total.item() < best_recon_loss:
        best_recon_loss = loss_total.item()
        best_Z = Z_opt.detach().clone()

    if (epoch + 1) % 100 == 0:
        print(f"   Recon {epoch + 1:3d}/300 | Total={loss_total.item():.6f} "
              f"Meas={loss_meas.item():.6f} ODE={loss_ode.item():.6f}")

# 最终提取重建视频 (确保输出严格符合物理动力学)
with torch.no_grad():
    Z_final = odeint_rk4(ode_func, best_Z[0], t_span)
    x_recon = torch.clamp(decoder(Z_final.unsqueeze(0)), 0, 1).squeeze(0).cpu().numpy()
# ==========================================
# 6. 可视化
# ==========================================
x_gt = x_test_gt.cpu().numpy()

x_rec_t = torch.from_numpy(x_recon).float().to(device)
y_final = torch.einsum('tmhw,thw->tm', A, x_rec_t) / SCALE_FACTOR
meas_mse = nn.MSELoss()(y_final, y_measured).item()
img_mse = nn.MSELoss()(x_rec_t, x_test_gt).item()
print(f"\nFinal Results:")
print(f"   Measurement MSE = {meas_mse:.6f}")
print(f"   Image MSE       = {img_mse:.6f}")

frames = [0, T // 4, T // 2, 3 * T // 4, T - 1]
fig, axes = plt.subplots(2, len(frames), figsize=(15, 6))
for i, f in enumerate(frames):
    axes[0, i].imshow(x_gt[f], cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f"GT t={f}");
    axes[0, i].axis('off')
    axes[1, i].imshow(x_recon[f], cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title(f"Recon t={f}");
    axes[1, i].axis('off')
plt.suptitle(f"Hollow Ring SPI-NODE (SPF={SPF * 100:.0f}%)", fontsize=14)
plt.tight_layout()
plt.savefig("reconstruction_v2.png", dpi=150)
plt.show()
