import math

import torch


def compute_phi(x, y, angle=0.0, eps=1e-12):
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    radius = torch.sqrt(torch.clamp(x**2 + y**2, min=eps))
    inside = radius - (x * direction_x + y * direction_y)
    return torch.sqrt(torch.clamp(0.5 * inside, min=eps))


def compute_phi_derivs(x, y, angle=0.0, eps=1e-12):
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    radius = torch.sqrt(torch.clamp(x**2 + y**2, min=eps))
    inside = torch.clamp(radius - (x * direction_x + y * direction_y), min=eps)
    sqrt_inside = torch.sqrt(inside)
    coef = math.sqrt(2.0) / 4.0

    phi_x = coef * (x / radius - direction_x) / sqrt_inside
    phi_y = coef * (y / radius - direction_y) / sqrt_inside
    return phi_x.view(-1, 1), phi_y.view(-1, 1)


def first_derivs_static(model, x, y, angle=0.0):
    phi = compute_phi(x, y, angle=angle)
    inputs = torch.cat([x, y, phi], dim=1).requires_grad_(True)
    u_value = model(inputs).view(-1, 1)
    grads = torch.autograd.grad(outputs=u_value.sum(), inputs=inputs, create_graph=True)[0]
    return u_value, inputs, grads[:, 0:1], grads[:, 1:2], grads[:, 2:3]


def first_derivs_time(model, x, y, t, angle=0.0):
    phi = compute_phi(x, y, angle=angle)
    inputs = torch.cat([x, y, phi, t], dim=1).requires_grad_(True)
    u_value = model(inputs).view(-1, 1)
    grads = torch.autograd.grad(outputs=u_value.sum(), inputs=inputs, create_graph=True)[0]
    return u_value, inputs, grads[:, 0:1], grads[:, 1:2], grads[:, 2:3], grads[:, 3:4]


def second_partials_static(inputs, u_x, u_y, u_z):
    grad_ux = torch.autograd.grad(outputs=u_x.sum(), inputs=inputs, create_graph=True)[0]
    grad_uy = torch.autograd.grad(outputs=u_y.sum(), inputs=inputs, create_graph=True)[0]
    grad_uz = torch.autograd.grad(outputs=u_z.sum(), inputs=inputs, create_graph=True)[0]

    u_xx = grad_ux[:, 0:1]
    u_yy = grad_uy[:, 1:2]
    u_xz = grad_ux[:, 2:3]
    u_yz = grad_uy[:, 2:3]
    u_zz = grad_uz[:, 2:3]
    return u_xx, u_yy, u_xz, u_yz, u_zz


def laplace_from_feature_space(model, x, y, angle=0.0, eps=1e-12):
    phi_x, phi_y = compute_phi_derivs(x, y, angle=angle, eps=eps)
    _, inputs, u_x, u_y, u_z = first_derivs_static(model, x, y, angle=angle)
    u_xx, u_yy, u_xz, u_yz, u_zz = second_partials_static(inputs, u_x, u_y, u_z)
    radius = torch.sqrt(torch.clamp(x**2 + y**2, min=eps))
    return u_xx + u_yy + 2.0 * (u_xz * phi_x + u_yz * phi_y) + (u_zz / (4.0 * radius))

