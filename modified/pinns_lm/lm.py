import torch


def flatten_params(model):
    return torch.cat([param.view(-1) for param in model.parameters()])


def set_params(model, theta):
    start = 0
    for param in model.parameters():
        count = param.numel()
        param.data.copy_(theta[start:start + count].view_as(param))
        start += count


def compute_jacobian(residual, model):
    residual = residual.view(-1)
    theta = flatten_params(model)
    jacobian = torch.zeros(residual.numel(), theta.numel(), device=theta.device)

    for row in range(residual.numel()):
        grads = torch.autograd.grad(residual[row], model.parameters(), retain_graph=True)
        jacobian[row] = torch.cat([grad.view(-1) for grad in grads])

    return jacobian


def lm_step(model, residual_fn, lam):
    residual = residual_fn(model)
    loss = torch.dot(residual, residual)
    theta = flatten_params(model)
    jacobian = compute_jacobian(residual, model)
    system = jacobian.T @ jacobian + lam * torch.eye(theta.numel(), device=theta.device)
    delta = torch.linalg.solve(system, -jacobian.T @ residual)
    set_params(model, theta + delta)
    return loss.item(), theta, delta, residual.numel(), theta.numel()

