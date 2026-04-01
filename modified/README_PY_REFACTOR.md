# Python refactor

I kept the original notebooks and added a small reusable Python layout:

- `pinns_lm/networks.py`: shared `MLP`
- `pinns_lm/crack.py`: shared crack-feature mapping `phi`, derivatives, and Laplacian assembly
- `pinns_lm/lm.py`: shared LM helpers for the steady cases
- `pinns_lm/plotting.py`: shared grid generation, error evaluation, residual maps, and loss plots
- `run_f0_case.py`: steady crack problem with exact solution `u = phi`
- `run_f1_case.py`: steady crack problem with forcing `-Delta u = 1`
- `run_time_dep_case.py`: time-dependent heat equation case

Run examples:

```bash
python run_f0_case.py
python run_f1_case.py
python run_time_dep_case.py
```

Notes:

- The notebooks are still the reference versions for plotting and interactive work.
- The new scripts are meant to make the shared structure easier to read and reuse.
- I kept the sampling and LM logic close to the original notebooks instead of heavily redesigning the math.

Minimal example:

```python
from pinns_lm.plotting import (
    evaluate_static_residual,
    evaluate_static_solution,
    make_grid,
    plot_loss_history,
    plot_residual_maps,
    plot_solution_comparison,
)

xg, yg = make_grid(-1.0, 1.0, -1.0, 1.0, n=100, device=device)
result = evaluate_static_solution(model, xg, yg, exact_fn=exact_solution)
plot_solution_comparison(result)

residual_result = evaluate_static_residual(model, xg, yg, forcing=1.0)
plot_residual_maps(xg.cpu(), yg.cpu(), residual_result)
plot_loss_history(loss_history)
```
