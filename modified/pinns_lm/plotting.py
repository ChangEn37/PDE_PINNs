from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinns_lm.crack import compute_phi, laplace_from_feature_space


def make_grid(x_min, x_max, y_min, y_max, n=100, device=None):
    x = torch.linspace(x_min, x_max, n, device=device)
    y = torch.linspace(y_min, y_max, n, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    return x_grid, y_grid


def relative_error_metrics(pred, exact):
    pred = pred.detach().cpu().double()
    exact = exact.detach().cpu().double()
    diff = pred - exact
    exact_inf = torch.max(torch.abs(exact)).item()
    l2_rel = (torch.norm(diff) / torch.norm(exact)).item()
    inf_rel = (torch.norm(diff, p=float("inf")) / max(exact_inf, 1e-15)).item()
    pointwise_rel = torch.abs(diff) / max(exact_inf, 1e-15)
    return {
        "l2_relative": l2_rel,
        "linf_relative": inf_rel,
        "pointwise_relative": pointwise_rel,
    }


def evaluate_static_solution(model, x_grid, y_grid, exact_fn: Optional[Callable] = None):
    with torch.no_grad():
        phi = compute_phi(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1))
        inputs = torch.cat([x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), phi], dim=1)
        pred = model(inputs).reshape_as(x_grid)

        result = {
            "prediction": pred.detach().cpu(),
            "x_grid": x_grid.detach().cpu(),
            "y_grid": y_grid.detach().cpu(),
        }

        if exact_fn is not None:
            exact = exact_fn(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)).reshape_as(x_grid)
            metrics = relative_error_metrics(pred, exact)
            result["exact"] = exact.detach().cpu()
            result["pointwise_relative"] = metrics["pointwise_relative"].reshape_as(x_grid)
            result["metrics"] = metrics

        return result


def evaluate_static_residual(model, x_grid, y_grid, forcing=0.0):
    lap = laplace_from_feature_space(model, x_grid.reshape(-1, 1), y_grid.reshape(-1, 1))
    residual = -lap - forcing
    residual = residual.reshape_as(x_grid).detach().cpu()
    return {
        "residual": residual,
        "abs_residual": torch.abs(residual),
        "log_abs_residual": torch.log(torch.clamp(torch.abs(residual), min=1e-15)),
    }


def evaluate_time_solution(model, x_grid, y_grid, time_value, exact_fn: Optional[Callable] = None):
    t_grid = torch.full_like(x_grid.reshape(-1, 1), float(time_value))
    with torch.no_grad():
        phi = compute_phi(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1))
        inputs = torch.cat([x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), phi, t_grid], dim=1)
        pred = model(inputs).reshape_as(x_grid)

        result = {
            "prediction": pred.detach().cpu(),
            "x_grid": x_grid.detach().cpu(),
            "y_grid": y_grid.detach().cpu(),
            "t_value": float(time_value),
        }

        if exact_fn is not None:
            exact = exact_fn(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), t_grid).reshape_as(x_grid)
            metrics = relative_error_metrics(pred, exact)
            result["exact"] = exact.detach().cpu()
            result["pointwise_relative"] = metrics["pointwise_relative"].reshape_as(x_grid)
            result["metrics"] = metrics

        return result


def evaluate_custom_time_solution(model, x_grid, y_grid, time_value, feature_fn: Callable, exact_fn: Optional[Callable] = None):
    t_grid = torch.full_like(x_grid.reshape(-1, 1), float(time_value))
    with torch.no_grad():
        feature = feature_fn(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), t_grid)
        inputs = torch.cat([x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), feature, t_grid], dim=1)
        pred = model(inputs).reshape_as(x_grid)

        result = {
            "prediction": pred.detach().cpu(),
            "x_grid": x_grid.detach().cpu(),
            "y_grid": y_grid.detach().cpu(),
            "t_value": float(time_value),
        }

        if exact_fn is not None:
            exact = exact_fn(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), t_grid).reshape_as(x_grid)
            metrics = relative_error_metrics(pred, exact)
            result["exact"] = exact.detach().cpu()
            result["pointwise_relative"] = metrics["pointwise_relative"].reshape_as(x_grid)
            result["metrics"] = metrics

        return result


def evaluate_time_residual(residual_fn: Callable, x_grid, y_grid, time_value):
    t_grid = torch.full_like(x_grid.reshape(-1, 1), float(time_value))
    residual = residual_fn(
        x_grid.reshape(-1, 1),
        y_grid.reshape(-1, 1),
        t_grid,
    ).reshape_as(x_grid).detach().cpu()
    return {
        "residual": residual,
        "abs_residual": torch.abs(residual),
        "log_abs_residual": torch.log(torch.clamp(torch.abs(residual), min=1e-15)),
        "t_value": float(time_value),
    }


def plot_solution_comparison(result, titles=("PINN Prediction", "Exact Solution", "Relative Error"), cmap_pred="viridis", cmap_err="hot"):
    x_np = result["x_grid"].numpy()
    y_np = result["y_grid"].numpy()
    pred_np = result["prediction"].numpy()

    if "exact" not in result or "pointwise_relative" not in result:
        raise ValueError("result must include exact and pointwise_relative for comparison plots")

    exact_np = result["exact"].numpy()
    err_np = result["pointwise_relative"].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    image0 = axes[0].contourf(x_np, y_np, pred_np, 50, cmap=cmap_pred)
    fig.colorbar(image0, ax=axes[0])
    axes[0].set_title(titles[0])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    image1 = axes[1].contourf(x_np, y_np, exact_np, 50, cmap=cmap_pred)
    fig.colorbar(image1, ax=axes[1])
    axes[1].set_title(titles[1])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    image2 = axes[2].contourf(x_np, y_np, err_np, 50, cmap=cmap_err)
    fig.colorbar(image2, ax=axes[2])
    axes[2].set_title(titles[2])
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.tight_layout()
    return fig, axes


def plot_residual_maps(x_grid, y_grid, residual_result, titles=("|Residual|", "log|Residual|"), cmap="viridis"):
    x_np = x_grid.detach().cpu().numpy()
    y_np = y_grid.detach().cpu().numpy()
    abs_np = residual_result["abs_residual"].numpy()
    log_np = residual_result["log_abs_residual"].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    image0 = axes[0].contourf(x_np, y_np, abs_np, 50, cmap=cmap)
    fig.colorbar(image0, ax=axes[0])
    axes[0].set_title(titles[0])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    image1 = axes[1].contourf(x_np, y_np, log_np, 50, cmap=cmap)
    fig.colorbar(image1, ax=axes[1])
    axes[1].set_title(titles[1])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    fig.tight_layout()
    return fig, axes


def plot_loss_history(loss_history, title="LM training loss"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.arange(len(loss_history)), loss_history, lw=2)
    ax.set_xlabel("LM iteration")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_prediction_views(result, title_prefix="Prediction", cmap="viridis", zlim=None):
    x_np = result["x_grid"].numpy()
    y_np = result["y_grid"].numpy()
    pred_np = result["prediction"].numpy()
    t_value = result.get("t_value", None)
    time_suffix = f", t={t_value:.2f}" if t_value is not None else ""

    fig = plt.figure(figsize=(18, 5))

    ax0 = fig.add_subplot(131)
    image0 = ax0.contourf(x_np, y_np, pred_np, 50, cmap=cmap)
    fig.colorbar(image0, ax=ax0)
    ax0.set_title(f"{title_prefix}{time_suffix}")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_aspect("equal")

    ax1 = fig.add_subplot(132, projection="3d")
    ax1.plot_surface(x_np, y_np, pred_np, cmap=cmap, edgecolor="none")
    ax1.set_title(f"Surface{time_suffix}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    if zlim is not None:
        ax1.set_zlim(*zlim)

    ax2 = fig.add_subplot(133, projection="3d")
    ax2.contour3D(x_np, y_np, pred_np, 50, cmap=cmap)
    ax2.set_title(f"3D contours{time_suffix}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    if zlim is not None:
        ax2.set_zlim(*zlim)

    fig.tight_layout()
    return fig, (ax0, ax1, ax2)
