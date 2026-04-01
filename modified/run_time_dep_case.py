import math
import json
from pathlib import Path

import numpy as np
import torch

from pinns_lm.crack import compute_phi, compute_phi_derivs, first_derivs_static, first_derivs_time, second_partials_static
from pinns_lm.networks import MLP
from pinns_lm.plotting import (
    evaluate_time_residual,
    evaluate_time_solution,
    make_grid,
    plot_loss_history,
    plot_residual_maps,
    plot_solution_comparison,
)
from pinns_lm.sampling import (
    sample_time_collocation,
    sample_time_crack_boundary,
    sample_time_initial_slice,
    sample_time_outer_boundary,
    sample_time_tip_and_crack,
)


torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 0.25
N_B_OUTER = 20
N_F = 4 * N_B_OUTER
N_B_CRACK = N_B_OUTER
N_TIP = N_B_OUTER
N_IC = N_B_OUTER
EPOCHS = 3000
LAM = 10000
LAM_UP = 3.0
LAM_DOWN = 0.7
MAX_TRIAL = 1
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "time_dep"


def exact_solution(x, y, t=None):
    phi = compute_phi(x, y)
    return (x**2 + y**2) * phi

def sample_ic(count):
    x, y, t = sample_time_initial_slice(count, X_MIN, X_MAX, Y_MIN, Y_MAX, device)
    phi = compute_phi(x, y)
    inputs = torch.cat([x, y, phi, t], dim=1)
    return inputs, exact_solution(x, y)


def compute_time_residual(model, x, y, t):
    phi = compute_phi(x, y)
    phi_x, phi_y = compute_phi_derivs(x, y)
    _, inputs, u_x, u_y, u_z, u_t = first_derivs_time(model, x, y, t)
    u_xx, u_yy, u_xz, u_yz, u_zz = second_partials_static(inputs, u_x, u_y, u_z)
    radius = torch.sqrt(torch.clamp(x**2 + y**2, min=1e-12))
    lap = u_xx + u_yy + 2.0 * (u_xz * phi_x + u_yz * phi_y) + (u_zz / (4.0 * radius))
    residual = u_t - lap + 6.0 * phi
    return residual, u_zz, u_xz, u_yz


def sample_all():
    x_f, y_f, t_f = sample_time_collocation(N_F, X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX, device)
    x_b_outer, y_b_outer, t_b_outer = sample_time_outer_boundary(N_B_OUTER, X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX, device)
    x_b_crack, y_b_crack, t_b_crack = sample_time_crack_boundary(N_B_CRACK, T_MIN, T_MAX, device)

    x_b = torch.cat([x_b_outer, x_b_crack], dim=0)
    y_b = torch.cat([y_b_outer, y_b_crack], dim=0)
    t_b = torch.cat([t_b_outer, t_b_crack], dim=0)
    phi_b = compute_phi(x_b, y_b)
    g_b = exact_solution(x_b, y_b)
    inp_bc = torch.cat([x_b, y_b, phi_b, t_b], dim=1)

    x_crack, y_crack, t_crack, x_tip, y_tip, t_tip = sample_time_tip_and_crack(N_TIP, N_B_CRACK, T_MIN, T_MAX, device)
    inp_ic, g_ic = sample_ic(N_IC)

    return {
        "x_f": x_f,
        "y_f": y_f,
        "t_f": t_f,
        "inp_bc": inp_bc,
        "g_b": g_b,
        "x_crack": x_crack,
        "y_crack": y_crack,
        "t_crack": t_crack,
        "x_tip": x_tip,
        "y_tip": y_tip,
        "t_tip": t_tip,
        "inp_ic": inp_ic,
        "g_ic": g_ic,
    }


def residual_vector(model, data):
    r_f, _, _, _ = compute_time_residual(model, data["x_f"], data["y_f"], data["t_f"])
    r_bc = model(data["inp_bc"]) - data["g_b"]
    r_ic = model(data["inp_ic"]) - data["g_ic"]
    _, u_zz_tip, _, _ = compute_time_residual(model, data["x_tip"], data["y_tip"], data["t_tip"])

    return torch.cat([r_f.view(-1), r_bc.view(-1), r_ic.view(-1), u_zz_tip.view(-1)], dim=0)


def flatten_params(model):
    return torch.cat([param.view(-1) for param in model.parameters()])


def compute_jacobian(residual, model):
    residual = residual.view(-1)
    theta = flatten_params(model)
    jacobian = torch.zeros(residual.numel(), theta.numel(), device=theta.device)

    for row in range(residual.numel()):
        grads = torch.autograd.grad(residual[row], model.parameters(), retain_graph=True)
        jacobian[row] = torch.cat([grad.view(-1) for grad in grads])

    return jacobian


def lm_step_adaptive(model, data, lam):
    params = list(model.parameters())
    theta0 = [param.data.clone() for param in params]
    residual0 = residual_vector(model, data)
    loss0 = 0.5 * torch.sum(residual0**2)
    jacobian = compute_jacobian(residual0, model)
    system = jacobian.T @ jacobian + lam * torch.eye(jacobian.shape[1], device=device)
    gradient = jacobian.T @ residual0.detach()
    delta = torch.linalg.solve(system, -gradient)

    for _ in range(MAX_TRIAL):
        start = 0
        for param, param0 in zip(params, theta0):
            count = param.numel()
            param.data = param0 + delta[start:start + count].view_as(param)
            start += count

        residual1 = residual_vector(model, data)
        loss1 = 0.5 * torch.sum(residual1**2)
        if loss1 < loss0:
            return max(lam * LAM_DOWN, 1e-9), True, torch.sqrt(2.0 * loss1).item()

        for param, param0 in zip(params, theta0):
            param.data = param0
        lam = min(lam * LAM_UP, 1e8)

    return lam, False, torch.sqrt(2.0 * loss0).item()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = MLP([4, 30, 1], last_bias=False).to(device)
    data = sample_all()
    lam = LAM
    loss_history = []

    for epoch in range(EPOCHS):
        lam, ok, residual_norm = lm_step_adaptive(model, data, lam)
        loss = residual_norm**2
        loss_history.append(loss)
        if epoch % 10 == 0:
            status = "accept" if ok else "reject"
            print(f"LM {epoch:04d} | loss {loss:.3e} | lambda={lam:.1e} | {status}")

    np.savez(RESULTS_DIR / "time_dep_lm_loss.npz", epoch=np.arange(len(loss_history)), loss=loss_history)
    np.savez(RESULTS_DIR / "time_dep_loss.npz", epoch=np.arange(len(loss_history)), loss=loss_history)

    x_grid, y_grid = make_grid(X_MIN, X_MAX, Y_MIN, Y_MAX, n=100, device=device)
    eval_time = 0.05
    solution_result = evaluate_time_solution(model, x_grid, y_grid, time_value=eval_time, exact_fn=exact_solution)
    residual_result = evaluate_time_residual(
        lambda x, y, t: compute_time_residual(model, x, y, t)[0],
        x_grid,
        y_grid,
        time_value=eval_time,
    )

    metrics = solution_result["metrics"]
    print(f"t = {eval_time:.3f} | relative L2 error = {metrics['l2_relative']:.6e}")
    print(f"t = {eval_time:.3f} | relative Linf error = {metrics['linf_relative']:.6e}")

    with open(RESULTS_DIR / "time_dep_metrics_t005.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "t_value": eval_time,
                "l2_relative": metrics["l2_relative"],
                "linf_relative": metrics["linf_relative"],
            },
            file,
            indent=2,
        )

    fig_solution, _ = plot_solution_comparison(solution_result)
    fig_solution.savefig(RESULTS_DIR / "time_dep_solution_t005.png", dpi=200, bbox_inches="tight")
    fig_solution.savefig(RESULTS_DIR / "time_dep_solution.png", dpi=200, bbox_inches="tight")

    fig_residual, _ = plot_residual_maps(x_grid, y_grid, residual_result)
    fig_residual.savefig(RESULTS_DIR / "time_dep_residual_t005.png", dpi=200, bbox_inches="tight")
    fig_residual.savefig(RESULTS_DIR / "time_dep_residual.png", dpi=200, bbox_inches="tight")

    fig_loss, _ = plot_loss_history(loss_history, title="time-dependent LM training loss")
    fig_loss.savefig(RESULTS_DIR / "time_dep_loss_history.png", dpi=200, bbox_inches="tight")
    fig_loss.savefig(RESULTS_DIR / "time_dep_loss.png", dpi=200, bbox_inches="tight")

    with open(RESULTS_DIR / "time_dep_metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "t_value": eval_time,
                "l2_relative": metrics["l2_relative"],
                "linf_relative": metrics["linf_relative"],
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
