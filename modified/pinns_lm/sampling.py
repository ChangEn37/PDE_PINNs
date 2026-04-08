import math

import torch


def sample_static_collocation(count, x_min, x_max, y_min, y_max, device, crack_tol=1e-8):
    x = torch.rand(count * 2, 1) * (x_max - x_min) + x_min
    y = torch.rand(count * 2, 1) * (y_max - y_min) + y_min
    mask = ~((torch.abs(y) < crack_tol) & (x >= 0))
    index = mask.squeeze(1)
    x = x[index][:count].view(-1, 1)
    y = y[index][:count].view(-1, 1)
    return x.to(device), y.to(device)


def sample_square_boundary(count, x_min, x_max, y_min, y_max, device):
    line_x = torch.linspace(x_min, x_max, count).view(-1, 1)
    line_y = torch.linspace(y_min, y_max, count).view(-1, 1)
    xb = torch.cat([line_x, line_x, torch.full_like(line_y, x_min), torch.full_like(line_y, x_max)])
    yb = torch.cat([torch.full_like(line_x, y_min), torch.full_like(line_x, y_max), line_y, line_y])
    return xb.to(device), yb.to(device)


def sample_tip_and_crack(n_tip, n_crack, device, crack_start=1e-4, crack_end=1.0, tip_radius=1e-3):
    x_crack = torch.linspace(crack_start, crack_end, n_crack, device=device).unsqueeze(1)
    y_crack = torch.zeros_like(x_crack)

    theta = torch.linspace(0, 2 * math.pi, n_tip, device=device)
    radius = tip_radius * torch.ones_like(theta)
    x_tip = (radius * torch.cos(theta)).view(-1, 1)
    y_tip = (radius * torch.sin(theta)).view(-1, 1)
    return x_crack, y_crack, x_tip, y_tip


def sample_time_collocation(count, x_min, x_max, y_min, y_max, t_min, t_max, device, crack_tol=1e-8):
    x = torch.rand(count * 2, 1) * (x_max - x_min) + x_min
    y = torch.rand(count * 2, 1) * (y_max - y_min) + y_min
    t = torch.rand(count * 2, 1) * (t_max - t_min) + t_min
    mask = (torch.abs(y) >= crack_tol).squeeze(1)
    x = x[mask][:count].view(-1, 1)
    y = y[mask][:count].view(-1, 1)
    t = t[mask][:count].view(-1, 1)
    return x.to(device), y.to(device), t.to(device)


def sample_time_outer_boundary(count, x_min, x_max, y_min, y_max, t_min, t_max, device):
    x_left = torch.full((count, 1), x_min)
    y_left = torch.linspace(y_min, y_max, count).unsqueeze(1)
    x_right = torch.full((count, 1), x_max)
    y_right = torch.linspace(y_min, y_max, count).unsqueeze(1)
    y_bottom = torch.full((count, 1), y_min)
    x_bottom = torch.linspace(x_min, x_max, count).unsqueeze(1)
    y_top = torch.full((count, 1), y_max)
    x_top = torch.linspace(x_min, x_max, count).unsqueeze(1)

    x_all = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
    y_all = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)
    t_all = torch.rand_like(x_all) * (t_max - t_min) + t_min
    return x_all.to(device), y_all.to(device), t_all.to(device)


def sample_time_crack_boundary(count, t_min, t_max, device, crack_end=1.0):
    t = torch.rand(count, 1, device=device) * (t_max - t_min) + t_min
    x_vals = torch.linspace(0, crack_end, count, device=device).unsqueeze(1)
    y_vals = torch.zeros_like(x_vals)
    return x_vals, y_vals, t


def sample_time_tip_and_crack(n_tip, n_crack, t_min, t_max, device, crack_start=1e-4, crack_end=1.0, tip_radius=1e-3):
    x_crack, y_crack, x_tip, y_tip = sample_tip_and_crack(
        n_tip=n_tip,
        n_crack=n_crack,
        device=device,
        crack_start=crack_start,
        crack_end=crack_end,
        tip_radius=tip_radius,
    )
    t_crack = torch.rand(n_crack, 1, device=device) * (t_max - t_min) + t_min
    t_tip = torch.rand(n_tip, 1, device=device) * (t_max - t_min) + t_min
    return x_crack, y_crack, t_crack, x_tip, y_tip, t_tip


def sample_time_initial_slice(count, x_min, x_max, y_min, y_max, device, crack_tol=1e-8):
    x = torch.rand(count * 2, 1) * (x_max - x_min) + x_min
    y = torch.rand(count * 2, 1) * (y_max - y_min) + y_min
    t = torch.zeros_like(x)
    mask = (torch.abs(y) >= crack_tol).squeeze(1)
    x = x[mask][:count].view(-1, 1)
    y = y[mask][:count].view(-1, 1)
    t = t[mask][:count].view(-1, 1)
    return x.to(device), y.to(device), t.to(device)


def sample_moving_interface_collocation(count, x_min, x_max, y_min, y_max, t_min, t_max, alpha_fn, device, crack_tol=1e-8):
    x = torch.rand(count * 2, 1) * (x_max - x_min) + x_min
    y = torch.rand(count * 2, 1) * (y_max - y_min) + y_min
    t = torch.rand(count * 2, 1) * (t_max - t_min) + t_min
    alpha_value = alpha_fn(t)
    dist_to_crack = torch.abs(y - alpha_value * x) / torch.sqrt(1.0 + alpha_value**2)
    mask = (dist_to_crack >= crack_tol).squeeze(1)
    x = x[mask][:count].view(-1, 1)
    y = y[mask][:count].view(-1, 1)
    t = t[mask][:count].view(-1, 1)
    return x.to(device), y.to(device), t.to(device)


def sample_moving_interface_boundary(count, t_min, t_max, alpha_fn, device, crack_end=1.0):
    t = torch.rand(count, 1, device=device) * (t_max - t_min) + t_min
    x_vals = torch.linspace(0, crack_end, count, device=device).unsqueeze(1)
    y_vals = x_vals * alpha_fn(t)
    return x_vals, y_vals, t
