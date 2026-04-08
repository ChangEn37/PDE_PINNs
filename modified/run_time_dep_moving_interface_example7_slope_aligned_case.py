from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pinns_lm.moving_interface_slope import (
    alpha,
    compute_crack_gradient_loss,
    compute_crack_uz_loss,
    compute_phi,
    compute_residual_time,
    compute_tip_constraint_loss,
)
from pinns_lm.plotting import (
    evaluate_custom_time_solution,
    evaluate_time_residual,
    make_grid,
    plot_loss_history,
    plot_prediction_views,
    plot_residual_maps,
)
from pinns_lm.sampling import (
    sample_moving_interface_boundary,
    sample_moving_interface_collocation,
    sample_square_boundary,
    sample_time_initial_slice,
    sample_time_outer_boundary,
)


torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 0.25

N_F = 1000
N_B_OUTER = 1000
N_B_CRACK = 1000
N_IC_INTERIOR = 1000
N_IC_BOUNDARY = 1000

EPOCHS = 3000
RESAMPLE_EVERY = 50
LOG_EVERY = 10

LAMBDA_BC = 1.0
LAMBDA_IC = 1.0
LAMBDA_TIP = 1.0
LAMBDA_CRACK_GRAD = 1.0
LAMBDA_CRACK_UZ = 0.0

LBFGS_LR = 1.0
LBFGS_MAX_ITER = 20
LBFGS_HISTORY_SIZE = 50

EVAL_TIME = 0.20
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "time_dep_moving_interface_example7_slope_aligned"


class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = torch.tanh
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        value = inputs
        for layer in self.layers[:-1]:
            value = self.activation(layer(value))
        return self.layers[-1](value)


def build_feature_inputs(x, y, t):
    phi = compute_phi(x, y, t)
    return torch.cat([x, y, phi, t], dim=1)


def sample_dataset():
    x_f, y_f, t_f = sample_moving_interface_collocation(
        N_F, X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX, alpha, device
    )
    x_b_outer, y_b_outer, t_b_outer = sample_time_outer_boundary(
        N_B_OUTER, X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX, device
    )
    x_b_crack, y_b_crack, t_b_crack = sample_moving_interface_boundary(N_B_CRACK, T_MIN, T_MAX, alpha, device)

    x_b = torch.cat([x_b_outer, x_b_crack], dim=0)
    y_b = torch.cat([y_b_outer, y_b_crack], dim=0)
    t_b = torch.cat([t_b_outer, t_b_crack], dim=0)
    inp_bc = build_feature_inputs(x_b, y_b, t_b)
    g_b = torch.zeros_like(x_b)

    x_ic_f, y_ic_f, t_ic_f = sample_time_initial_slice(N_IC_INTERIOR, X_MIN, X_MAX, Y_MIN, Y_MAX, device)
    x_ic_out, y_ic_out = sample_square_boundary(N_IC_BOUNDARY, X_MIN, X_MAX, Y_MIN, Y_MAX, device)
    t_ic_out = torch.zeros_like(x_ic_out)
    x_ic = torch.cat([x_ic_f, x_ic_out], dim=0)
    y_ic = torch.cat([y_ic_f, y_ic_out], dim=0)
    t_ic = torch.cat([t_ic_f, t_ic_out], dim=0)
    inp_ic = build_feature_inputs(x_ic, y_ic, t_ic)
    u0 = torch.zeros_like(x_ic)

    return {
        "x_f": x_f,
        "y_f": y_f,
        "t_f": t_f,
        "x_b_crack": x_b_crack,
        "y_b_crack": y_b_crack,
        "t_b_crack": t_b_crack,
        "inp_bc": inp_bc,
        "g_b": g_b,
        "inp_ic": inp_ic,
        "u0": u0,
    }


def compute_loss_terms(model, data):
    residual_f, _, _, _ = compute_residual_time(model, data["x_f"], data["y_f"], data["t_f"], forcing=1.0)
    loss_pde = torch.mean(residual_f**2)
    loss_bc = torch.mean((model(data["inp_bc"]) - data["g_b"]) ** 2)
    loss_ic = torch.mean((model(data["inp_ic"]) - data["u0"]) ** 2)
    loss_tip = compute_tip_constraint_loss(model, data["t_b_crack"])
    loss_crack_grad = compute_crack_gradient_loss(model, data["x_b_crack"], data["y_b_crack"], data["t_b_crack"])
    loss_crack_uz = compute_crack_uz_loss(model, data["x_b_crack"], data["y_b_crack"], data["t_b_crack"])

    total = (
        loss_pde
        + LAMBDA_BC * loss_bc
        + LAMBDA_IC * loss_ic
        + LAMBDA_TIP * loss_tip
        + LAMBDA_CRACK_GRAD * loss_crack_grad
        + LAMBDA_CRACK_UZ * loss_crack_uz
    )
    return total, {
        "pde": loss_pde.item(),
        "bc": loss_bc.item(),
        "ic": loss_ic.item(),
        "tip": loss_tip.item(),
        "crack_grad": loss_crack_grad.item(),
        "crack_uz": loss_crack_uz.item(),
    }


def train_model(model):
    train_data = sample_dataset()
    val_data = sample_dataset()

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    loss_history = []
    val_history = []
    val_epochs = []

    for epoch in range(EPOCHS):
        def closure():
            optimizer.zero_grad()
            total_loss, _ = compute_loss_terms(model, train_data)
            total_loss.backward()
            return total_loss

        optimizer.step(closure)
        train_snapshot, _ = compute_loss_terms(model, train_data)
        loss_history.append(train_snapshot.item())

        if epoch % LOG_EVERY == 0 or epoch == EPOCHS - 1:
            train_snapshot, train_terms = compute_loss_terms(model, train_data)
            val_loss, val_terms = compute_loss_terms(model, val_data)
            val_history.append(val_loss.item())
            val_epochs.append(epoch)
            print(
                f"Epoch {epoch:04d} | "
                f"TrainLoss={train_snapshot.item():.3e} "
                f"(pde={train_terms['pde']:.2e}, bc={train_terms['bc']:.2e}, ic={train_terms['ic']:.2e}, "
                f"tip={train_terms['tip']:.2e}, crack_grad={train_terms['crack_grad']:.2e}) | "
                f"ValLoss={val_loss.item():.3e} "
                f"(pde={val_terms['pde']:.2e}, bc={val_terms['bc']:.2e}, ic={val_terms['ic']:.2e}, "
                f"tip={val_terms['tip']:.2e}, crack_grad={val_terms['crack_grad']:.2e})"
            )

        if RESAMPLE_EVERY and epoch % RESAMPLE_EVERY == 0 and epoch > 0:
            print(f"[Resample] epoch {epoch}: regenerating train points...")
            train_data = sample_dataset()

    return loss_history, val_history, val_epochs


def evaluate_model(model, loss_history, val_history, val_epochs):
    x_grid, y_grid = make_grid(X_MIN, X_MAX, Y_MIN, Y_MAX, n=100, device=device)
    prediction_result = evaluate_custom_time_solution(
        model,
        x_grid,
        y_grid,
        time_value=EVAL_TIME,
        feature_fn=compute_phi,
    )
    residual_result = evaluate_time_residual(
        lambda x, y, t: compute_residual_time(model, x, y, t, forcing=1.0)[0],
        x_grid,
        y_grid,
        time_value=EVAL_TIME,
    )

    np.savez(
        RESULTS_DIR / "time_dep_moving_example7_slope_aligned_loss.npz",
        epoch=np.arange(len(loss_history)),
        loss=loss_history,
    )
    np.savez(
        RESULTS_DIR / "time_dep_moving_example7_slope_aligned_val_loss.npz",
        epoch=np.array(val_epochs),
        loss=val_history,
    )

    fig_pred, _ = plot_prediction_views(
        prediction_result,
        title_prefix="Example 7 slope-aligned prediction",
        zlim=(0.0, 0.2),
    )
    fig_pred.savefig(
        RESULTS_DIR / "time_dep_moving_example7_slope_aligned_prediction.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig_res, _ = plot_residual_maps(
        x_grid,
        y_grid,
        residual_result,
        titles=(
            f"Example 7 slope-aligned |Residual| at t={EVAL_TIME:.2f}",
            f"Example 7 slope-aligned log|Residual| at t={EVAL_TIME:.2f}",
        ),
    )
    fig_res.savefig(
        RESULTS_DIR / "time_dep_moving_example7_slope_aligned_residual.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig_loss, ax_loss = plot_loss_history(loss_history, title="Example 7 slope-aligned training loss")
    val_epochs_arr = np.array(val_epochs)
    if len(val_epochs_arr) > 0:
        ax_loss.semilogy(val_epochs_arr, val_history, label="Validation Loss")
        ax_loss.legend()
    fig_loss.savefig(
        RESULTS_DIR / "time_dep_moving_example7_slope_aligned_loss.png",
        dpi=200,
        bbox_inches="tight",
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = MLP([4, 64, 64, 1]).to(device)
    loss_history, val_history, val_epochs = train_model(model)
    evaluate_model(model, loss_history, val_history, val_epochs)


if __name__ == "__main__":
    main()
