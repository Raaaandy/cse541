import datetime
from json import load
from multiprocessing.pool import RUN
from turtle import shape
from matplotlib.tri import Triangulation
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from tqdm import tqdm
import torch.nn.functional as F
import argparse

class Exp3Bandit:
    def __init__(self, arms, gamma=0.1):
        """
        arms: list of tuples, e.g., [(pde_co, bc_co, mse_co), ...]
        gamma: 探索率 (0, 1]
        """
        self.arms = arms
        self.num_arms = len(arms)
        self.gamma = gamma
        self.weights = np.ones(self.num_arms)
        self.probabilities = np.ones(self.num_arms) / self.num_arms

    def select_arm(self):
        # 根据 Exp3 算法计算当前选择各摇臂的概率
        sum_weights = np.sum(self.weights)
        self.probabilities = (1.0 - self.gamma) * (self.weights / sum_weights) + (self.gamma / self.num_arms)
        
        # 根据概率分布采样一个摇臂
        chosen_arm_index = np.random.choice(self.num_arms, p=self.probabilities)
        return chosen_arm_index, self.arms[chosen_arm_index]

    def update(self, chosen_arm_index, reward):
        # 计算估计奖励
        estimated_reward = reward / self.probabilities[chosen_arm_index]
        
        # 更新该摇臂的权重 (为防止溢出，可以对指数部分进行裁剪)
        exponent = (self.gamma * estimated_reward) / self.num_arms
        exponent = np.clip(exponent, -10, 10) 
        self.weights[chosen_arm_index] *= np.exp(exponent)

class FourierFeatureMapping(nn.Module):
    """
    Fourier transformer.
    """
    def __init__(self, input_dim, mapping_size, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size

        B = torch.randn(input_dim, mapping_size // 2) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        # (N, input_dim) @ (input_dim, mapping_size/2) -> (N, mapping_size/2)
        x_proj = x @ self.B * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# FCN by Tingying
class FCN(nn.Module):
    def __init__(self,layers, uvp_mean=None, uvp_std=None,
                 fourier_mapping_size=None, fourier_scale=None, U_lid = 2, use_conv=False):
        super().__init__()
        self.fourier_mapping = None
        self.U_lid = U_lid
        self.use_conv = use_conv
        
        if fourier_mapping_size and fourier_scale:
            input_dim = layers[0]
            self.fourier_mapping = FourierFeatureMapping(input_dim, fourier_mapping_size, fourier_scale)
            layers[0] = fourier_mapping_size

        # self.activation = nn.Tanh()
        # self.activation = nn.SiLU()
        self.activation = nn.GELU()
        self.loss_function = nn.MSELoss(reduction='sum')
        
        # if use_conv and fourier_mapping_size:
        #     # 添加1D卷积层处理Fourier特征
        #     self.conv_layers = nn.Sequential(
        #         nn.Conv1d(1, 32, kernel_size=3, padding=1),
        #         nn.GELU(),
        #         nn.Conv1d(32, 64, kernel_size=3, padding=1),
        #         nn.GELU(),
        #         nn.AdaptiveAvgPool1d(layers[1])  # 调整到第二层维度
        #     )
        #     layers[0] = layers[1]  # 卷积后的维度
        
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)
        
        if uvp_mean is not None and uvp_std is not None:
            self.register_buffer('uvp_mean', torch.tensor(uvp_mean, dtype=torch.float32))
            self.register_buffer('uvp_std', torch.tensor(uvp_std, dtype=torch.float32))
        else:
            self.uvp_mean = None
            self.uvp_std = None

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = x.to(next(self.parameters()).device).float()
        # a = x.float()
        if self.fourier_mapping is not None:
            a = self.fourier_mapping(a)
            
        # 如果启用卷积，对Fourier特征进行1D卷积处理
        if self.use_conv and hasattr(self, 'conv_layers'):
            # 将Fourier特征reshape为1D卷积输入: (batch, 1, features)
            a = a.unsqueeze(1)  # (N, 1, fourier_features)
            a = self.conv_layers(a)  # (N, 64, reduced_features)
            a = a.mean(dim=1)  # (N, reduced_features)
            
        residual = None
        for i in range(len(self.linears)-1):
            z = self.linears[i](a)
            a = self.activation(z)
            # 添加残差连接，每3层一次
            if i == 2:  # 第3层保存residual
                residual = a
            elif i == 5 and residual is not None:  # 第6层添加residual
                a = a + residual
        a = self.linears[-1](a)
        # g_phys = self.g_uv(x)
        # g_norm = (g_phys - self.uvp_mean[0:2]) / self.uvp_std[0:2] if self.uvp_mean is not None and self.uvp_std is not None else g_phys

        # l = self.l_xy(x)
        # uvp_norm_hat = g_norm + l * a[:, 0:2]
        # p_hat = a[:, 2:3]
        # a = torch.cat([uvp_norm_hat, p_hat], dim=1)
        return a

    def l_xy(self, x):
    # x: (N,2) (x, y)
        return (x[:,0:1] * (1 - x[:,0:1]) *
                x[:,1:2] * (1 - x[:,1:2]))  # 形状 (N,1)
    
    def g_uv(self, x, eps=1e-3):
        """
        输入 x: (N,2) 张量
        返回 g: (N,3) 张量，对应 [u_bc, v_bc, p_bc]
        """
        y = x[:,1:2]
        top_mask = (y > 1 - eps).float()    # 只要 y 非常接近 1，就当作在顶壁
        profile = y**2 * (3-2*y) * self.U_lid
        u_bc =  top_mask * profile # 平滑过渡
        v_bc = torch.zeros_like(u_bc)       # 整个边界 v=0
        return torch.cat([u_bc, v_bc], dim=1)  # (N,2)


    def denormalize(self, data_normalized):
        """
        Denormalize the output data.
        """
        if self.uvp_mean is not None and self.uvp_std is not None:
            return data_normalized * self.uvp_std + self.uvp_mean
        else:
            return data_normalized
        
    def predict(self, x):
        pred = self.forward(x)
        return self.denormalize(pred)
    
    def _compute_normalized_loss(self, pred_normalized, y_physical):
        """
        Helper function to compute MSE loss in the normalized space.
        It normalizes the physical target `y_physical` before comparing.
        """
        if not torch.is_tensor(y_physical):
            y_physical = torch.tensor(y_physical, dtype=pred_normalized.dtype, device=pred_normalized.device)
            
        if self.uvp_mean is not None and self.uvp_std is not None:
            y_normalized = (y_physical - self.uvp_mean) / self.uvp_std
            return self.loss_function(pred_normalized, y_normalized)
        else:
            # If no normalization params, compare directly
            return self.loss_function(pred_normalized, y_physical)

    def lossG(self, x_G, y_G):
        pred_G = self.forward(x_G)
        loss_G = self.loss_function(pred_G, y_G)
        return loss_G

    def lossBC(self, x_BC, y_BC):
        pred_BC = self.predict(x_BC)  # Only p component
        loss_BC = self.loss_function(pred_BC[:,0:2], y_BC[:,0:2])  # Compare only p component
        return loss_BC
    
    def lossCorner(self, x_corner, y_corner):

        pred_corner = self.predict(x_corner)
        loss_corner = self.loss_function(pred_corner, y_corner)
        return loss_corner

    def lossMSE(self, x, y):
        pred = self.predict(x)[:,0:2]
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=pred.dtype, device=pred.device)

        # return torch.sqrt(torch.sum(pred - y) ** 2) / torch.sqrt(torch.sum(y ** 2))  # Normalized MSE loss
        return self.loss_function(pred, y[:, 0:2])  # Compare only u and v components


    def lossPDE(self, x_PDE, rho=1.0, nu=1/100, con_weight=1, x_weight=1, y_weight=1):
        # 减少PDE计算点数来避免OOM

        g = x_PDE.clone().detach().to(next(self.parameters()).device)
        g.requires_grad = True  # Enable differentiation

        output = self.predict(g)
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        # 计算 u, v, p 对 x, y 的梯度（求梯度是对Denormalized后的物理量求导）
        u_grad = autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0]
        u_x = u_grad[:, 0:1]
        u_y = u_grad[:, 1:2]
        v_grad = autograd.grad(v, g, torch.ones_like(v), create_graph=True)[0]
        v_x = v_grad[:, 0:1]
        v_y = v_grad[:, 1:2]
        p_grad = autograd.grad(p, g, torch.ones_like(p), create_graph=True)[0]
        p_x = p_grad[:, 0:1]
        p_y = p_grad[:, 1:2]

        u_xx = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = autograd.grad(u_y, g, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        v_xx = autograd.grad(v_x, g, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = autograd.grad(v_y, g, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

        continuity = u_x + v_y
        continuity_loss = self.loss_function(continuity, torch.zeros_like(continuity))

        x_momentum = u * u_x + v * u_y + (1 / rho) * p_x - nu * (u_xx + u_yy)
        x_momentum_loss = self.loss_function(x_momentum, torch.zeros_like(x_momentum))

        y_momentum = u * v_x + v * v_y + (1 / rho) * p_y - nu * (v_xx + v_yy)
        y_momentum_loss = self.loss_function(y_momentum, torch.zeros_like(y_momentum))


        total_loss = con_weight * continuity_loss + x_weight * x_momentum_loss + \
            y_weight * y_momentum_loss
        return total_loss

    def lossPDE_info(self, x_PDE, rho=1.0, nu=1/100, con_weight=1, x_weight=1, y_weight=1):
        g = x_PDE.clone()
        g.requires_grad = True  # Enable differentiation

        output = self.predict(g)
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        # 计算 u, v, p 对 x, y 的梯度（注意：这里求梯度仍是对反归一化后的物理量求导）
        u_grad = autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0]
        u_x = u_grad[:, 0:1]
        u_y = u_grad[:, 1:2]
        v_grad = autograd.grad(v, g, torch.ones_like(v), create_graph=True)[0]
        v_x = v_grad[:, 0:1]
        v_y = v_grad[:, 1:2]
        p_grad = autograd.grad(p, g, torch.ones_like(p), create_graph=True)[0]
        p_x = p_grad[:, 0:1]
        p_y = p_grad[:, 1:2]

        u_xx = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = autograd.grad(u_y, g, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        v_xx = autograd.grad(v_x, g, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
        v_yy = autograd.grad(v_y, g, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]
        vals = torch.cat([g[:, 0:1], g[:, 1:2], u, v, p, u_x, u_y, v_x, v_y, p_x, p_y, u_xx, u_yy, v_xx, v_yy], dim=1)
        np.savetxt("vals.txt", vals.detach().cpu().numpy(), header="x y u v p u_x u_y v_x v_y p_x p_y u_xx u_yy v_xx v_yy", fmt="%f")
        print("vals.txt saved")
        continuity = u_x + v_y
        continuity_loss = F.mse_loss(continuity, torch.zeros_like(continuity), reduction='none')

        x_momentum = u * u_x + v * u_y + (1 / rho) * p_x - nu * (u_xx + u_yy)
        x_momentum_loss = F.mse_loss(x_momentum, torch.zeros_like(x_momentum), reduction='none')

        y_momentum = u * v_x + v * v_y + (1 / rho) * p_y - nu * (v_xx + v_yy)
        y_momentum_loss = F.mse_loss(y_momentum, torch.zeros_like(y_momentum), reduction='none')

        total_loss = continuity_loss + x_momentum_loss + y_momentum_loss
        return total_loss, continuity_loss, x_momentum_loss, y_momentum_loss


    # total loss
    def loss(self, x_PDE, y_true, X_BC, Y_BC, X_global, Y_global, mse_co, pde_co, bc_co):
        loss_pde = self.lossPDE(x_PDE, x_weight=1, y_weight=1)
        loss_mse = self.lossMSE(X_global, Y_global)
        loss_bc = self.lossBC(X_BC, Y_BC)

        # loss_corner = self.lossCorner(x_corner, y_corner)
        # total_loss = 1*loss_bc + 1*loss_pde + 2*loss_g + 1*loss_corner
        # total_loss = 10 * loss_mse + 20 * loss_bc
        # total_loss = 10 * loss_pde
        # total_loss = 5 * loss_pde + 10 * loss_bc + 5 * loss_G 


        total_loss = pde_co * loss_pde + bc_co * loss_bc + mse_co * loss_mse

        # total_loss = pde_co * loss_pde + mse_co * loss_bc + mse_co * loss_mse

        # total_loss = 10 * loss_pde
        # total_loss = loss_pde + loss_bc +  loss_G
        # return loss_bc, loss_pde, loss_g, total_loss, loss_corner
        return loss_pde, loss_mse, loss_bc, total_loss

    # Optimizer closure function
    def closure(self, optimizer, X_train_Nu, Y_train_Nu, X_train_Nf, X_train_G, Y_train_G, X_train_corner, Y_train_corner):
        optimizer.zero_grad()

        # Unpack individual losses
        # loss_bc, loss_pde, loss_g, total_loss, loss_corner = self.loss(X_train_Nu, Y_train_Nu, X_train_Nf, X_train_G, Y_train_G, X_train_corner, Y_train_corner)
        loss_pde, total_loss = self.loss(X_train_Nu, Y_train_Nu, X_train_Nf, X_train_G, Y_train_G, X_train_corner, Y_train_corner)
        # Backpropagation with total_loss
        total_loss.backward()

        # Increment iteration count and print progress every 100 iterations
        self.iter += 1
        if self.iter % 100 == 0:
            with torch.no_grad():
                print(f"Iteration {self.iter}: total_loss = {total_loss.item()}")

        return total_loss

def load_pure_txt(filepath):
    print(f"Loading data from: {filepath}")
    return np.loadtxt(filepath)


def generate_bc_points(test_data):
    x = test_data[:, 0:2]
    y_true = test_data[:, 2:5]
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y_true, dtype=torch.float32)
    epsilon = 1e-6

#   is_left = abs(x_tensor[:, 0] - (-0.5)) < epsilon
#   is_right = abs(x_tensor[:, 0] - 0.5) < epsilon
#   is_bottom = abs(x_tensor[:, 1] - (-0.5)) < epsilon
#   is_top = abs(x_tensor[:, 1] - 0.5) < epsilon

    is_left = x_tensor[:, 0] < epsilon
    is_right = x_tensor[:, 0] > (1 - epsilon)
    is_bottom = x_tensor[:, 1] < epsilon
    is_top = x_tensor[:, 1] > (1 - epsilon)

    is_boundary = is_left | is_right | is_bottom | is_top

    boundary_indices = torch.nonzero(is_boundary).squeeze()
    interior_indices = torch.nonzero(~is_boundary).squeeze()

    if len(boundary_indices.shape) == 0:
        boundary_indices = boundary_indices.unsqueeze(0)
    if len(interior_indices.shape) == 0:
        interior_indices = interior_indices.unsqueeze(0)


    X_BC = x_tensor[boundary_indices]
    Y_BC = y_tensor[boundary_indices]
    X_interior = x_tensor[interior_indices]

    print(f"边界点数量: {X_BC.shape[0]}")
    print(f"内部点数量: {X_interior.shape[0]}")

    return X_BC, Y_BC, X_interior


def test_model(model, test_data, rho=1.0, nu=1/100, output_file="results.txt", pde_info=False, U_lid=4):
    """
    return:predictions, true_values, abs_x, abs_y, test_data_x, test_data_y, predictions_x, predictions_y
    """
    x = test_data[:, 0:2]

    x_tensor = torch.tensor(x, dtype=torch.float32, device = device)
    mask_x = torch.isclose(x_tensor[:,0], torch.tensor(0.5, device=device))
    indices_x = torch.argsort(x_tensor[mask_x][:,1])
    x_tensor_x = x_tensor[mask_x][indices_x]

    mask_y = torch.isclose(x_tensor[:,1], torch.tensor(0.5, device=device))
    indices_y = torch.argsort(x_tensor[mask_y][:,0])
    x_tensor_y = x_tensor[mask_y][indices_y]

    mask_wall = (
         np.isclose(x[:,0], 0.0, atol=1e-3) |
         np.isclose(x[:,0], 1.0, atol=1e-3) |
         np.isclose(x[:,1], 0.0, atol=1e-3) |
         np.isclose(x[:,1], 1.0, atol=1e-3)
    )
    mask_top = np.isclose(x[:,1], 1.0, atol=1e-3)

    with torch.no_grad():
        # predictions = model.predict(x_tensor).numpy() # u, v, p
        # predictions_x = model.predict(x_tensor_x).numpy()
        # predictions_y = model.predict(x_tensor_y).numpy()
        predictions = model.forward(x_tensor) # u, v, p
        predictions_x = model.forward(x_tensor_x)
        predictions_y = model.forward(x_tensor_y)
        predictions_denorm = model.predict(x_tensor) # u, v, p
    predictions = predictions.cpu().numpy()
    predictions_x = predictions_x.cpu().numpy()
    predictions_y = predictions_y.cpu().numpy()
    predictions_denorm = predictions_denorm.cpu().numpy()

    predictions_denorm[mask_wall, 0] = 0.0
    predictions_denorm[mask_wall, 1] = 0.0
    # 顶壁 u=U_lid
    predictions_denorm[mask_top, 0] = U_lid
    with open(output_file, 'w') as f:
        for i in range(len(x)):
            line = f"{x[i, 0]:<25} {x[i, 1]:<25} {predictions_denorm[i, 0]:<25.5f} {predictions_denorm[i, 1]:<25.5f} {predictions_denorm[i, 2]:<25.5f}\n"
            f.write(line)
    print(f"results write to {output_file}")

    mask_x = np.isclose(test_data[:, 0], 0.5)
    subset_x = test_data[mask_x]

    # 2. 根据第二列 (y 坐标) 进行排序
    # subset_x[:, 1] 是 y 坐标列
    sort_idx_x = np.argsort(subset_x[:, 1])

    # 3. 得到排序后的完整数据（前两列坐标，后三列物理量）
    test_data_x_sorted = subset_x[sort_idx_x]

    # 4. 如果你只需要 u, v, p 三列：
    test_data_x = test_data_x_sorted[:, 2:5]

    mask_y = np.isclose(test_data[:, 1], 0.5)
    subset_y = test_data[mask_y]

    # 2. 根据第一列 (x 坐标) 进行排序
    sort_idx_y = np.argsort(subset_y[:, 0])

    # 3. 得到排序后的数据
    test_data_y_sorted = subset_y[sort_idx_y]

    # 4. 提取 u, v, p 三列
    test_data_y = test_data_y_sorted[:, 2:5]

    true_values = test_data[:, 2:5]  # u, v, p

    abs_x = np.abs(predictions_x - test_data_x)
    abs_y = np.abs(predictions_y - test_data_y)

    if pde_info:
        print(f"abs x=200 per component (u, v, p): {abs_x.mean(axis=0)}")
        print(f"abs y=200 per component (u, v, p): {abs_y.mean(axis=0)}")

        X_BC, Y_BC, X_interior = generate_bc_points(test_data)
        print(f'X interior shape: {X_interior.shape}')
        pde_loss, con, x, y = model.lossPDE_info(X_interior)

        coords = X_interior.detach().cpu().numpy()
        pde_loss = pde_loss.detach().cpu().numpy().reshape(-1, 1)
        con = con.detach().cpu().numpy().reshape(-1, 1)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        data_con = np.hstack((coords, con))
        np.savetxt("con_test_0626.txt", data_con)
        data_x = np.hstack((coords, x))
        np.savetxt("x_0626.txt", data_x)
        data_y = np.hstack((coords, y))
        np.savetxt("y_0626.txt", data_y)
        data = np.hstack((coords, pde_loss))
        np.savetxt("pde_loss.txt", data)

        with torch.no_grad():
            bc_loss = model.lossBC(X_BC, Y_BC).item()

        print(f"PDE Loss: {np.mean(pde_loss):.6f}")
        print(f"BC Loss: {bc_loss:.6f}")


    return x, predictions, true_values, abs_x, abs_y, test_data_x, test_data_y, predictions_x, predictions_y

def visualize_results(test_data, predictions, save_path=None):
    x = test_data[:, 0]
    y = test_data[:, 1]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    components = ['u', 'v', 'p']
    titles = ['Velocity u', 'Velocity v', 'Pressure p']

    for i, (comp, title) in enumerate(zip(components, titles)):
        true_values = test_data[:, i+2]
        pred_values = predictions[:, i]

        vmin = min(np.min(true_values), np.min(pred_values))
        vmax = max(np.max(true_values), np.max(pred_values))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        sc1 = axes[i, 0].scatter(x, y, c=true_values, cmap='viridis', norm=norm)
        axes[i, 0].set_title(f'True {title}')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        plt.colorbar(sc1, ax=axes[i, 0])

        sc2 = axes[i, 1].scatter(x, y, c=pred_values, cmap='viridis', norm=norm)
        axes[i, 1].set_title(f'Predicted {title}')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        plt.colorbar(sc2, ax=axes[i, 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_error_distribution(true_values, predictions):
    import seaborn as sns
    errors = predictions - true_values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = ['u', 'v', 'p']

    for i, comp in enumerate(components):
        sns.histplot(errors[:, i], kde=True, ax=axes[i])
        axes[i].set_title(f'Error Distribution for {comp}')
        axes[i].set_xlabel(f'Error in {comp}')

    plt.tight_layout()
    plt.show()

def visualize_direct_comparison(test_data, predictions, save_path=None):

    x = test_data[:, 0]
    y = test_data[:, 1]
    u_true = test_data[:, 2]
    v_true = test_data[:, 3]
    p_true = test_data[:, 4]

    u_pred = predictions[:, 0]
    v_pred = predictions[:, 1]
    p_pred = predictions[:, 2]

    u_error = np.abs(u_pred - u_true)
    v_error = np.abs(v_pred - v_true)
    p_error = np.abs(p_pred - p_true)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    components = ['Velocity u', 'Velocity v', 'Pressure p']
    true_values = [u_true, v_true, p_true]
    pred_values = [u_pred, v_pred, p_pred]
    errors = [u_error, v_error, p_error]

    for i, (component, true, pred, error) in enumerate(zip(components, true_values, pred_values, errors)):
        vmin = min(np.min(true), np.min(pred))
        vmax = max(np.max(true), np.max(pred))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        sc1 = axes[i, 0].scatter(x, y, c=true, cmap='viridis', norm=norm)
        axes[i, 0].set_title(f'True {component}')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        plt.colorbar(sc1, ax=axes[i, 0])

        sc2 = axes[i, 1].scatter(x, y, c=pred, cmap='viridis', norm=norm)
        axes[i, 1].set_title(f'Predicted {component}')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        plt.colorbar(sc2, ax=axes[i, 1])

        sc3 = axes[i, 2].scatter(x, y, c=error, cmap='hot_r')
        axes[i, 2].set_title(f'Absolute Error in {component}')
        axes[i, 2].set_xlabel('x')
        axes[i, 2].set_ylabel('y')
        error_bar = plt.colorbar(sc3, ax=axes[i, 2])
        error_bar.set_label('|Pred - True|')

        max_err = np.max(error)
        mean_err = np.mean(error)
        median_err = np.median(error)
        axes[i, 2].text(0.05, 0.95,
                      f'Max: {max_err:.4f}\nMean: {mean_err:.4f}\nMedian: {median_err:.4f}',
                      transform=axes[i, 2].transAxes,
                      fontsize=10,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', alpha=0.2))

    plt.tight_layout()
    if save_path:
        comparison_path = save_path.replace('.png', '_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"comparision picture: {comparison_path}")

    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (component, true, error) in enumerate(zip(components, true_values, errors)):
        im = axes[i].scatter(x, y, c=true, cmap='viridis', alpha=0.7)
        plt.colorbar(im, ax=axes[i])

        high_error_mask = error > np.mean(error)
        if np.any(high_error_mask):
            err_points_x = x[high_error_mask]
            err_points_y = y[high_error_mask]
            err_values = error[high_error_mask]

            sc = axes[i].scatter(err_points_x, err_points_y,
                               c=err_values, cmap='Reds',
                               edgecolors='white', linewidths=0.5,
                               s=50, alpha=0.8)

            err_cbar = plt.colorbar(sc, ax=axes[i])
            err_cbar.set_label('Error Magnitude')

        axes[i].set_title(f'{component} with Error Overlay')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')

    plt.tight_layout()
    if save_path:
        overlay_path = save_path.replace('.png', '_error_overlay.png')
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
        print(f"overlay graph: {overlay_path}")

    plt.show()

    return {
        'u_error': {'max': np.max(u_error), 'mean': np.mean(u_error), 'median': np.median(u_error)},
        'v_error': {'max': np.max(v_error), 'mean': np.mean(v_error), 'median': np.median(v_error)},
        'p_error': {'max': np.max(p_error), 'mean': np.mean(p_error), 'median': np.median(p_error)}
    }






def train_model(model, test_data, sampling_data_HG, mse_co, pde_co, model_save_path, threshold=None, epochs=3000, lr=0.0005):
    x = test_data[:, 0:2]
    y_true = test_data[:, 2:5]
    x_hf = sampling_data_HG[:, 0:2]
    y_true_hf = sampling_data_HG[:, 2:5]
    x_hf_tensor = torch.tensor(x_hf, dtype=torch.float32, device=device)
    y_hf_tensor = torch.tensor(y_true_hf, dtype=torch.float32, device=device)
    # 转换为torch张量
    x_tensor = torch.tensor(x, dtype=torch.float32, device = device)
    y_tensor = torch.tensor(y_true, dtype=torch.float32, device=device)
    corner_mask = (((x_hf_tensor[:,0] >= 7/8) | (x_hf_tensor[:,0] <= 1/8)) & (x_hf_tensor[:,1] >= 7/8))

    N = x_tensor.shape[0]
    tol = 1/40
    target = 0.5
    # mask for -0.5 ~ 0.5
    # mask = (x_tensor[:, 0] - tol <= -1*target) | (x_tensor[:, 0] + tol >= target) | \
    #    (x_tensor[..., 1] - tol <= -1*target) | (x_tensor[:, 1] + tol >= target)
    mask = (x_hf_tensor[:,0]  <= tol) |  (x_hf_tensor[:,0] >= 1 - tol) | \
              (x_hf_tensor[:,1] <= tol) | (x_hf_tensor[:,1] >= 1 - tol)
    # mask = (x_tensor[:,0]  <= tol) |  (x_tensor[:,0] >= 1 - tol) | \
    #           (x_tensor[:,1] <= tol) | (x_tensor[:,1] >= 1 - tol)

    corner_mask = corner_mask | mask
    # X_train_G = x_tensor[~mask]
    # y_true = y_tensor[~mask]  # 训练值
    X_train_G = x_tensor
    y_true = y_tensor
    boundary_points = x_hf_tensor[mask].to(device)
    boundary_values = y_hf_tensor[mask].to(device)

    print("X boundary shape", boundary_points.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                       mode='min', 
                                                       factor=0.5, 
                                                       patience=800)
    weight_arms = [
        (1.0,  60.0, 10.0),  # 偏重边界和数据
        (10.0, 60.0, 1.0),   # 偏重 PDE 和边界
        (10.0, 10.0, 10.0),  # 权重平均
        (20.0, 80.0, 20.0),  # 整体高惩罚
        (0.1,  60.0, 20.0)   # 早期可能更需要拟合数据和边界，弱化 PDE

    ]
    bandit = Exp3Bandit(arms=weight_arms, gamma=0.15)
    
    prev_loss = None
    best_loss = float('inf')
    final_pde_co = pde_co
    initial_pde_co = 1e-4 # Start with a very small weight for the PDE loss
    annealing_epochs = epochs // 2 # Number of epochs to gradually increase the PDE weight
    grad_clip_value = 1.0
    output_path = model_save_path.replace('.pth', '_training_log.txt')
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        # 2. 老虎机选择当前 Epoch 的权重组合
        arm_idx, (pde_w, bc_w, mse_w) = bandit.select_arm()

        # 3. 计算 Loss 
        loss_pde, loss_mse, boundary_loss, total_loss = model.loss(
            X_train_G, y_true, boundary_points, boundary_values, 
            x_tensor, y_tensor, 
            mse_co=mse_w, pde_co=pde_w, bc_co=bc_w
        )

        total_loss.backward()
        optimizer.step()

        # 4. 计算 Reward 并更新 Bandit
        current_loss_val = loss_pde.item() + loss_mse.item() + boundary_loss.item()
        if prev_loss is not None:
            # 奖励定义：Loss 下降的比例。如果 Loss 上升，reward 为负。
            # 这里加入了极小值 1e-8 防止除以 0
            reward = (prev_loss - current_loss_val) / (prev_loss + 1e-8)
            
            # 将 Reward 缩放/裁剪到合理范围 [-1, 1] 防止权重爆炸
            reward = np.clip(reward * 100, -1.0, 1.0) 
            
            # 更新摇臂权重
            bandit.update(arm_idx, reward)
            
        prev_loss = current_loss_val

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: total_loss = {current_loss_val:.6f}, "
                  f"Chosen Arm = {arm_idx} ({pde_w}, {bc_w}, {mse_w}), "
                  f"Arm Probs = {[round(p, 3) for p in bandit.probabilities]}")


    return model

def plot_velocity_field_with_arrows(test_data, predictions, uvp_mean=None, uvp_std=None, 
                                    sample_ratio=0.01,
                                    save_path="pinn_results_hg_stream.jpg", u_lid=None):

    import matplotlib.pyplot as plt
    import numpy as np
    if uvp_mean is None and uvp_std is None:
        predictions_physical = predictions.copy()
    else:
        uvp_mean = np.array(uvp_mean)
        uvp_std = np.array(uvp_std)
        predictions_physical = predictions.copy()
        predictions_physical[:, 0] = predictions[:, 0] * uvp_std[0] + uvp_mean[0]  # u
        predictions_physical[:, 1] = predictions[:, 1] * uvp_std[1] + uvp_mean[1]  # v

    x = test_data[:, 0]
    y = test_data[:, 1]
    u = predictions_physical[:, 0]
    v = predictions_physical[:, 1]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    eps = 1e-6
    # 顶边
    top = np.abs(y - y_max) < eps
    if u_lid is not None:
        u[top] = u_lid
    v[top] = 0
    # 其余三边
    bottom = np.abs(y - y_min) < eps
    left   = np.abs(x - x_min) < eps
    right  = np.abs(x - x_max) < eps
    wall = bottom | left | right
    u[wall] = 0
    v[wall] = 0
    order = np.lexsort((x, y))
    x_s, y_s = x[order], y[order]
    u_s, v_s = u[order], v[order]
    x_unique = np.unique(x_s)
    y_unique = np.unique(y_s)
    nx, ny = len(x_unique), len(y_unique)
    
    # Create a grid for the velocity field
    X = x_s.reshape(ny, nx)
    Y = y_s.reshape(ny, nx)
    U = u_s.reshape(ny, nx)
    V = v_s.reshape(ny, nx)
    Xg = X[1:-1, 1:-1]
    Yg = Y[1:-1, 1:-1]
    Ug = U[1:-1, 1:-1]
    Vg = V[1:-1, 1:-1]
    plt.figure(figsize=(8, 8))
    plt.streamplot(Xg, Yg, Ug, Vg, color='b', linewidth=0.5, density=2, arrowstyle='->')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Velocity Field Streamlines')
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.savefig(save_path)
    plt.close()
    



parser = argparse.ArgumentParser(
    description="Train or evaluate a PINN model with specified parameters."
)
parser.add_argument('--pde_info', action='store_true', 
                    help='Set this flag to print PDE information during evaluation.')
parser.add_argument('--Re', type=int, default=1000, 
                    help='Reynolds number (default: 1000)')
parser.add_argument('--U', type=int, default=10, 
                    help='Inlet velocity (default: 10)')
parser.add_argument('--do_training', action='store_true', 
                    help='Set this flag to train a new model.')
parser.add_argument('--num', type=int, default=None, 
                    help='Grid size (NX). If not set, defaults to 41 for training and 401 for evaluation.')

parser.add_argument('--dataset_base_path', type=str, 
                    default='/projects/bfth/rhe4/PINN/data_pinn',
                    help='Base path for datasets.')

parser.add_argument('--output_base_path', type=str, 
                    default='/projects/bfth/rhe4/PINN/results_pinn_only',
                    help='Base path for all outputs (models, plots, logs).')
parser.add_argument('--continue_training', action='store_true',
                    help='Set this flag to load and continue training an existing model.')
parser.add_argument('--training_epochs', type=int, default=3000,
                    help='Number of epochs for training (default: 3000).')
parser.add_argument('--eval_mse_model', action='store_true',
                    help='Set this flag to evaluate the mse model.')
args = parser.parse_args()
DO_TRAINING = args.do_training  # Set to True to train a new model, False to evaluate an existing one.

# --- Parameters for the Run ---
# These are used to name folders and select data
Re = args.Re
U = args.U
# Grid size (`num` or `NX`) is determined by the training/evaluation mode below
if args.num is not None:
    num = args.num
elif DO_TRAINING:
    # num = 41
    num = 81
else:
    # When evaluating, we typically test against the high-resolution dataset
    num = 401
    # num = 41


# --- Path Configuration ---
DATASET_BASE_PATH = args.dataset_base_path
OUTPUT_BASE_PATH = args.output_base_path
CONTINUE_TRAINING = args.continue_training

print("--- Running with the following configuration ---")
print(f"Mode: {'Training' if DO_TRAINING else 'Evaluation'}")
print(f"Continue Training: {CONTINUE_TRAINING}")
print(f"Reynolds Number (Re): {Re}")
print(f"Inlet Velocity (U): {U}")
print(f"Grid Size (num): {num}")
print(f"Dataset Path: {DATASET_BASE_PATH}")
print(f"Output Path: {OUTPUT_BASE_PATH}")
# --- Model Loading Configuration ---

# --- Automatic Folder & Path Generation ---
# Generate a unique name for this specific run
mode_str = "train" if DO_TRAINING else "eval"
run_folder_name = f"{mode_str}_Re{Re}_U{U}_NX{num}_CT{CONTINUE_TRAINING}"

# Create the main output directory for this run
RUN_OUTPUT_DIR = os.path.join(OUTPUT_BASE_PATH, run_folder_name)
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
print(f"============================================================")
print(f"Starting Run: {run_folder_name}")
print(f"All outputs will be saved to: {RUN_OUTPUT_DIR}")
print(f"============================================================\n")

# Define model save path for this run
model_save_path = os.path.join(RUN_OUTPUT_DIR, f"model_Re{Re}_U{U}_NX{num}.pth")
load_folder_name = f"train_Re{Re}_U{U}_NX{num}_CTFalse"  # Example folder name to load from
# Define model load path (if specified)
model_load_path = None
if DO_TRAINING and CONTINUE_TRAINING:
    # Find the corresponding model file in the specified run folder
    potential_path = os.path.join(OUTPUT_BASE_PATH, load_folder_name)
    model_files = [f for f in os.listdir(potential_path) if f.endswith('.pth')]
    if model_files:
        model_load_path = os.path.join(potential_path, model_files[0])
        model_save_path = os.path.join(RUN_OUTPUT_DIR, f"CT_model_Re{Re}_U{U}_NX{num}.pth")
        print(f"Found model to load: {model_load_path}")
    else:
        print(f"[Warning] No .pth model file found in specified load directory: {potential_path}")
elif not DO_TRAINING:
    print("[Error] In evaluation mode (DO_TRAINING=False), but no model specified in RUN_ID_TO_LOAD.")
    # raise SystemExit("Cannot evaluate without a model to load.") # Uncomment to stop execution

# ==============================================================================
# 2. DATA LOADING & PREPARATION
# ==============================================================================
SUBFOLDER_PATH = f"U{U}"
DATA_FILE = f"processed_Re{Re}_U{U}_NX{num}.txt"
DATA_FILE_HF = f"processed_Re{Re}_U{U}_NX401.txt"

data_path = os.path.join(DATASET_BASE_PATH, SUBFOLDER_PATH, DATA_FILE)
data_path_hf = os.path.join(DATASET_BASE_PATH, SUBFOLDER_PATH, DATA_FILE_HF)
print(f"Loading data from: {data_path}")
print(f"Refined data from: {data_path_hf}")

test_data = load_pure_txt(data_path)
test_data_hf = load_pure_txt(data_path_hf)
print(f"Loaded {len(test_data)} data points\n")

rho = 1.0
nu = 1/100 # More general definition of nu based on Re

# Normalize data
normalize = True
if normalize:
    # if DO_TRAINING:
    #     uvp_mean = np.mean(test_data[:, 2:5], axis=0)
    #     uvp_std = np.std(test_data[:, 2:5], axis=0)
    # else:
    #     training_data_path = os.path.join(DATASET_BASE_PATH, SUBFOLDER_PATH, f"processed_Re{Re}_U{U}_NX41.txt")
    #     training_data = load_pure_txt(training_data_path)
    #     uvp_mean = np.mean(training_data[:, 2:5], axis=0)
    #     uvp_std = np.std(training_data[:, 2:5], axis=0)
    uvp_std = np.std(test_data[:, 2:5], axis=0)
    uvp_mean = np.mean(test_data[:, 2:5], axis=0)
    test_data_normalized = test_data.copy()
    print("Data normalized.")
else:
    uvp_mean = None
    uvp_std = None
    test_data_normalized = test_data.copy()
    print("Data not normalized.")

# ==============================================================================
# 3. MODEL INITIALIZATION & TRAINING / EVALUATION
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FCN([2, 256, 256, 256, 256, 256, 256, 3], uvp_mean=uvp_mean, uvp_std=uvp_std, 
            fourier_mapping_size=64, fourier_scale=3.0, U_lid=U, use_conv=False).to(device)
# model = FCN([2, 64, 64, 64, 64, 64, 3], uvp_mean=uvp_mean, uvp_std=uvp_std, 
#             fourier_mapping_size=256, fourier_scale=3.0)
if DO_TRAINING:
    print("--- Mode: Training ---")
    if CONTINUE_TRAINING and model_load_path and os.path.exists(model_load_path):
        try:
            model.load_state_dict(torch.load(model_load_path))
            print(f"Successfully loaded model from {model_load_path} to continue training.")
        except Exception as e:
            print(f"[Error] Failed to load model for continuation: {e}")
    else:
        if CONTINUE_TRAINING:
            print("[Warning] CONTINUE_TRAINING is True, but no valid model path was found. Starting from scratch.")
    # if not CONTINUE_TRAINING:
    #     print("Initializing a new model for training.")
    #     model = train_model(model, test_data=test_data_normalized, sampling_data_HG=test_data_hf, mse_co=20, pde_co=0, epochs=args.training_epochs, model_save_path=model_save_path, lr=0.0005)
    # else:
    #     print("Continuing training with the loaded model.")
    #     model = train_model(model, test_data=test_data_normalized, sampling_data_HG=test_data_hf, mse_co=20, pde_co=20, epochs=args.training_epochs, model_save_path=model_save_path, lr=1e-5)
    if not CONTINUE_TRAINING:
        print("Initializing a new model for training.")
        model = train_model(model, test_data=test_data_normalized, sampling_data_HG=test_data_normalized, mse_co=10, pde_co=20, epochs=args.training_epochs, model_save_path=model_save_path, lr=0.0005)
    else:
        print("Continuing training with the loaded model.")
        model = train_model(model, test_data=test_data_normalized, sampling_data_HG=test_data_normalized, mse_co=20, pde_co=20, epochs=args.training_epochs, model_save_path=model_save_path, lr=1e-5)

        # model = train_model(model, test_data=test_data_normalized, sampling_data_HG=high_gradient_area, mse_co=0, pde_co=10, threshold=0.5, max_epochs=50000, lr=0.0005)
    try:
        print(f"\nTraining complete. Model saved to: {model_save_path}")
    except Exception as e:
        print(f"[Error] Failed to save model: {e}")

else: # Evaluation mode
    print("--- Mode: Evaluation ---")

    eval_load_path = f"/projects/bfth/rhe4/PINN/results_pinn_only/train_Re1000_U10_NX81_CTFalse/model_Re1000_U10_NX81.pth"

    if eval_load_path and os.path.exists(eval_load_path):
        try:
            model.load_state_dict(torch.load(eval_load_path))
            print(f"Successfully loaded model for evaluation: {eval_load_path}")
        except Exception as e:
            print(f"Loading error: {e}")
            raise KeyboardInterrupt()
    else:
        print(f"[Error] Model file not found at expected path: {eval_load_path}")
        raise KeyboardInterrupt()

    # Perform testing/evaluation
output_file_path = os.path.join(RUN_OUTPUT_DIR, "pinn_results.txt")
x, predictions, true_values, abs_x, abs_y, test_data_x, test_data_y, prediction_x, prediction_y = test_model(model, test_data_normalized, output_file=output_file_path, pde_info=args.pde_info, rho=rho, nu=nu, U_lid=U)
normalize = True
print("Denormalizing predictions and test data... line 1106")
# test_data_x = test_data_x *  uvp_std + uvp_mean if normalize else test_data_x
# test_data_y = test_data_y *  uvp_std + uvp_mean if normalize else test_data_y
predictions_x = prediction_x * uvp_std + uvp_mean if normalize else prediction_x
predictions_y = prediction_y * uvp_std + uvp_mean if normalize else prediction_y
abs_x = np.abs(predictions_x - test_data_x)
abs_y = np.abs(predictions_y - test_data_y)
denormalized_predictions = predictions * uvp_std + uvp_mean if normalize else predictions
mse = np.mean((denormalized_predictions - test_data[:, 2:5]) ** 2)
print(f"\nEvaluation complete. MSE for grid {num}x{num}: {mse:.6e}")
# Save metrics to a file
with open(os.path.join(RUN_OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Run ID: {run_folder_name}\n")
    f.write(f"Evaluation against dataset: {data_path}\n")
    f.write(f"Mean Squared Error (denormalized): {mse:.6e}\n")


# ==============================================================================
# 4. VISUALIZATION & PLOTTING
# ==============================================================================
print("\n--- Generating Plots ---")
x_axis_41_401 = np.arange(81) * 5
print(test_data_x.shape, test_data_y.shape, predictions_x.shape, predictions_y.shape)
# Plot 1: Absolute Error at x=0.5
plt.figure(figsize=(10, 7))
# plt.plot(abs_x[:, 0], label='U Error (pred vs true)')
# plt.plot(test_data_x[:, 0], label=f'U True (Grid {num})', linestyle=':')
# plt.plot(predictions_x[:, 0], label='U Predicted', linestyle='--')
if not DO_TRAINING:
    true_value_41 = np.loadtxt(os.path.join(DATASET_BASE_PATH, SUBFOLDER_PATH, f"processed_Re{Re}_U{U}_NX401.txt"))
    # true_value_41 = np.loadtxt('/mmfs1/gscratch/intelligentsystems/randy/PINN/new_dataset_81/Re=900/processed_Re900_U9_NX81.txt')
    true_value_41 = true_value_41[np.argsort(true_value_41[:,1])]

    if num == 41:
        plt.plot(\
            true_value_41[np.isclose(true_value_41[:,0], 0.5)][:,2], \
                label='U True (Grid 41)', linestyle='-.')
    else:
        plt.plot(x_axis_41_401, abs_x[:, 0], label='U Error (pred vs true)')
        plt.plot(x_axis_41_401, test_data_x[:, 0], label=f'U True (Grid {num})', linestyle=':')
        plt.plot(x_axis_41_401, predictions_x[:, 0], label='U Predicted', linestyle='--')
        plt.plot(true_value_41[np.isclose(true_value_41[:,0], 0.5)][:,2], \
                label='U True (Grid 401)', linestyle='-.')
plt.xlabel('Index along Y-axis')
plt.ylabel('Velocity U')
plt.title(f'Velocity U along the line x=0.5')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(RUN_OUTPUT_DIR, 'plot_U_at_x_half.png'))
plt.close()
print("Plot 1 saved: plot_U_at_x_half.png")

# Plot 2: Absolute Error at y=0.5
plt.figure(figsize=(10, 7))

if not DO_TRAINING:
    true_value_41 = np.loadtxt(os.path.join(DATASET_BASE_PATH, SUBFOLDER_PATH, f"processed_Re{Re}_U{U}_NX401.txt"))
    # true_value_41 = np.loadtxt('/mmfs1/gscratch/intelligentsystems/randy/PINN/new_dataset_81/Re=900/processed_Re900_U9_NX81.txt')

    mask_y = np.isclose(true_value_41[:,1], 0.5)
    # y41 = (true_value_41[mask_y][:,3] - uvp_mean_41[1]) / uvp_std_41[1]
    true_value_41 = true_value_41[mask_y]
    order = np.argsort(true_value_41[:,0])
    y41 = true_value_41[order][:,3]
    x41 = x_axis_41_401
    if num == 41:
        plt.plot(\
            y41, \
                label='V True (Grid 41)', linestyle='-.')
    else:
        plt.plot(x41, abs_y[:, 1], label='V Error (pred vs true)')
        plt.plot(x41, test_data_y[:, 1], label=f'V True (Grid {num})', linestyle=':')
        plt.plot(x41, predictions_y[:, 1], label='V Predicted', linestyle='--')
        plt.plot(\
            y41,\
            label='V True (Grid 401)', linestyle='-.')
plt.xlabel('Index along X-axis')
plt.ylabel('Velocity V')
plt.title(f'Velocity V along the line y=0.5')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(RUN_OUTPUT_DIR, 'plot_V_at_y_half.png'))
plt.close()
print("Plot 2 saved: plot_V_at_y_half.png")

# Plot 3: Heatmap Visualization
results_path = os.path.join(RUN_OUTPUT_DIR, f"results_heatmap.png")
visualize_results(test_data, denormalized_predictions, save_path=results_path)

# Plot 4: Velocity Field Arrow Plot
arrow_plot_path = os.path.join(RUN_OUTPUT_DIR, f"velocity_field_arrows.jpg")
plot_velocity_field_with_arrows(test_data, denormalized_predictions, save_path=arrow_plot_path, sample_ratio=0.5, u_lid=U)

# Plot 5: Direct Comparison
comparison_path = os.path.join(RUN_OUTPUT_DIR, "direct_comparison.png")
error_stats = visualize_direct_comparison(test_data, denormalized_predictions, save_path=comparison_path)

print(f"All plots have been saved to the run directory: {RUN_OUTPUT_DIR}")

print(f"\n--- Run {run_folder_name} Finished ---")
