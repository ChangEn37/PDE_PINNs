from pathlib import Path

import numpy as np
import torch

from pinns_lm.crack import compute_phi, first_derivs_static, laplace_from_feature_space, second_partials_static
from pinns_lm.lm import lm_step, set_params
from pinns_lm.networks import MLP
from pinns_lm.plotting import evaluate_static_residual, make_grid, plot_loss_history, plot_residual_maps
from pinns_lm.sampling import sample_square_boundary, sample_static_collocation, sample_tip_and_crack


torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
N_F = 50
N_B = 50
N_B_CRACK = 50
N_TIP = 20
EPOCHS = 1000
LAM = 1e3
LAM_UP = 3.0
LAM_DOWN = 0.7
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "f1"

def build_residual_fn(x_f, y_f, x_b, y_b):
    def residual_fn(model):
        residuals = []
        lap = laplace_from_feature_space(model, x_f, y_f)
        residuals.append((-lap - 1.0).view(-1))

        phi_b = compute_phi(x_b, y_b)
        inputs_b = torch.cat([x_b, y_b, phi_b], dim=1)
        residuals.append(model(inputs_b).view(-1))

        x_crack, y_crack, x_tip, y_tip = sample_tip_and_crack(N_TIP, N_B_CRACK, device)
        _, inputs_tip, u_x_tip, u_y_tip, u_z_tip = first_derivs_static(model, x_tip, y_tip)
        _, _, u_xz_tip, u_yz_tip, u_zz_tip = second_partials_static(inputs_tip, u_x_tip, u_y_tip, u_z_tip)
        _, inputs_crack, u_x_crack, u_y_crack, u_z_crack = first_derivs_static(model, x_crack, y_crack)
        _, _, u_xz_crack, u_yz_crack, _ = second_partials_static(inputs_crack, u_x_crack, u_y_crack, u_z_crack)

        residuals += [u_zz_tip.view(-1), u_xz_tip.view(-1), u_yz_tip.view(-1)]
        residuals += [u_xz_crack.view(-1), u_yz_crack.view(-1)]
        return torch.cat(residuals)

    return residual_fn


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = MLP([3, 30, 30, 1], last_bias=True).to(device)
    x_f, y_f = sample_static_collocation(N_F, X_MIN, X_MAX, Y_MIN, Y_MAX, device)
    x_b, y_b = sample_square_boundary(N_B, X_MIN, X_MAX, Y_MIN, Y_MAX, device)
    residual_fn = build_residual_fn(x_f, y_f, x_b, y_b)
    lam = LAM
    loss_history = []

    for epoch in range(EPOCHS):
        loss0, theta_old, _, _, _ = lm_step(model, residual_fn, lam)
        loss1 = torch.dot(residual_fn(model), residual_fn(model)).item()

        if loss1 < loss0:
            lam = max(lam * LAM_DOWN, 1e-9)
            status = "accept"
        else:
            lam = min(lam * LAM_UP, 1e8)
            set_params(model, theta_old)
            status = "reject"

        loss_history.append(loss1)
        print(f"LM {epoch:04d} | loss {loss1:.3e} | lambda={lam:.1e} | {status}")

    np.savez(RESULTS_DIR / "f1_lm_loss.npz", epoch=np.arange(len(loss_history)), loss=loss_history)
    np.savez(RESULTS_DIR / "f1_loss.npz", epoch=np.arange(len(loss_history)), loss=loss_history)

    x_grid, y_grid = make_grid(X_MIN, X_MAX, Y_MIN, Y_MAX, n=100, device=device)
    residual_result = evaluate_static_residual(model, x_grid, y_grid, forcing=1.0)

    fig_residual, _ = plot_residual_maps(x_grid, y_grid, residual_result)
    fig_residual.savefig(RESULTS_DIR / "f1_residual_maps.png", dpi=200, bbox_inches="tight")
    fig_residual.savefig(RESULTS_DIR / "f1_residual.png", dpi=200, bbox_inches="tight")

    fig_loss, _ = plot_loss_history(loss_history, title="f1 LM training loss")
    fig_loss.savefig(RESULTS_DIR / "f1_loss_history.png", dpi=200, bbox_inches="tight")
    fig_loss.savefig(RESULTS_DIR / "f1_loss.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
