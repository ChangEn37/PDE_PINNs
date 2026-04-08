import math

import torch

from pinns_lm.crack import second_partials_static


def alpha(t):
    return (math.pi / 4.0) * torch.sin(2.0 * math.pi * t)


def alpha_t(t):
    return (math.pi**2 / 2.0) * torch.cos(2.0 * math.pi * t)


def crack_angle(t):
    return torch.atan(alpha(t))


def crack_angle_t(t):
    slope = alpha(t)
    return alpha_t(t) / (1.0 + slope**2)


def compute_phi(x, y, t, eps=1e-12):
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    t = t.view(-1, 1)
    angle = crack_angle(t)
    radius = torch.sqrt(torch.clamp(x**2 + y**2, min=eps))
    inside = torch.clamp(radius - (x * torch.cos(angle) + y * torch.sin(angle)), min=eps)
    return torch.sqrt(0.5 * inside)


def compute_phi_and_derivs(x, y, t, eps=1e-12):
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    t = t.view(-1, 1)
    angle = crack_angle(t)
    angle_t = crack_angle_t(t)
    radius = torch.sqrt(torch.clamp(x**2 + y**2, min=eps))
    inside = torch.clamp(radius - (x * torch.cos(angle) + y * torch.sin(angle)), min=eps)
    phi = torch.sqrt(0.5 * inside)
    sqrt_inside = torch.sqrt(inside)
    coeff = math.sqrt(2.0) / 4.0

    phi_x = coeff * (x / radius - torch.cos(angle)) / sqrt_inside
    phi_y = coeff * (y / radius - torch.sin(angle)) / sqrt_inside
    phi_t = coeff * (x * torch.sin(angle) - y * torch.cos(angle)) / sqrt_inside
    phi_t = phi_t * angle_t
    return phi, phi_x, phi_y, phi_t


def first_derivs_time(model, x, y, t):
    phi, phi_x, phi_y, phi_t = compute_phi_and_derivs(x, y, t)
    inputs = torch.cat([x.view(-1, 1), y.view(-1, 1), phi.view(-1, 1), t.view(-1, 1)], dim=1)
    inputs = inputs.requires_grad_(True)
    u_value = model(inputs).view(-1, 1)
    grads = torch.autograd.grad(outputs=u_value.sum(), inputs=inputs, create_graph=True, retain_graph=True)[0]
    return (
        u_value,
        inputs,
        grads[:, 0:1],
        grads[:, 1:2],
        grads[:, 2:3],
        grads[:, 3:4],
        phi,
        phi_x,
        phi_y,
        phi_t,
    )


def compute_residual_time(model, x, y, t, forcing=1.0):
    _, inputs, u_x, u_y, u_z, u_t, _, phi_x, phi_y, phi_t = first_derivs_time(model, x, y, t)
    u_xx, u_yy, u_xz, u_yz, u_zz = second_partials_static(inputs, u_x, u_y, u_z)
    radius = torch.sqrt(torch.clamp(x.view(-1, 1) ** 2 + y.view(-1, 1) ** 2, min=1e-12))
    laplace_u = u_xx + u_yy + 2.0 * (u_xz * phi_x + u_yz * phi_y) + (u_zz / (4.0 * radius))
    time_u = u_z * phi_t + u_t
    residual = time_u - laplace_u - forcing
    return residual, u_zz, u_xz, u_yz


def compute_tip_regularization_terms(model, t):
    t = t.view(-1, 1)
    x = torch.zeros_like(t)
    y = torch.zeros_like(t)
    _, inputs, _, _, u_z, _, _, _, _, _ = first_derivs_time(model, x, y, t)
    grad_uz = torch.autograd.grad(outputs=u_z.sum(), inputs=inputs, create_graph=True, retain_graph=True)[0]
    u_xz = grad_uz[:, 0:1]
    u_yz = grad_uz[:, 1:2]
    u_zz = grad_uz[:, 2:3]
    return u_zz, u_xz, u_yz


def compute_tip_constraint_loss(model, t):
    u_zz, u_xz, u_yz = compute_tip_regularization_terms(model, t)
    return torch.mean(u_zz**2 + u_xz**2 + u_yz**2)


def compute_crack_gradient_terms(model, x, y, t):
    _, inputs, _, _, u_z, _, _, _, _, _ = first_derivs_time(model, x, y, t)
    grad_uz = torch.autograd.grad(outputs=u_z.sum(), inputs=inputs, create_graph=True, retain_graph=True)[0]
    u_xz = grad_uz[:, 0:1]
    u_yz = grad_uz[:, 1:2]
    return u_xz, u_yz


def compute_crack_gradient_loss(model, x, y, t):
    u_xz, u_yz = compute_crack_gradient_terms(model, x, y, t)
    return torch.mean(u_xz**2 + u_yz**2)


def compute_crack_uz_terms(model, x, y, t):
    _, _, _, _, u_z, _, _, _, _, _ = first_derivs_time(model, x, y, t)
    return u_z


def compute_crack_uz_loss(model, x, y, t):
    u_z = compute_crack_uz_terms(model, x, y, t)
    return torch.mean(u_z**2)
