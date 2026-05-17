Great results! Before jumping to prediction, here's what your analysis tells us about **predictability**:

| Metric            | Value     | Implication                                                             |
| ----------------- | --------- | ----------------------------------------------------------------------- |
| **LLE ≈ 0.003**   | Near zero | Long prediction horizon possible (not exponentially diverging)          |
| **DET = 0.19**    | Low       | Trajectories don't repeat deterministically → hard to predict far ahead |
| **Lmax = 5**      | Short     | Reliable prediction window is roughly **~5 time steps**                 |
| **PC1+PC2 = 80%** | Good      | 2D projection captures most variance → low-dim structure exists         |

---

### Prediction strategies suited to your system

Given **edge-of-chaos + low DET + Lmax=5**, the best-fit approaches are:

1. **Method of Analogues (Lorenz)** — find past 5D states closest to the current one and use their future evolution as the prediction. Works well for recurrence-based systems. No model training needed.

2. **Local Linear Maps** — fit a linear map in the neighborhood of each state in phase space. Captures local dynamics without assuming global linearity.

3. **Vector Autoregression (VAR)** — exploits cross-dimension correlations across all 5 dims. Good for the quasi-periodic component.

4. **Prediction horizon estimation** — use the LLE to compute the maximum reliable forecast window: `T_max = (1/LLE) * ln(σ_noise / σ_initial)`.

---

Should I create a PR in **bisandr/Events** with a `predict_5d.py` module implementing all of these, with:

- Prediction plots (predicted vs actual for each method)
- Forecast horizon estimation from LLE
- Cross-validation error across methods so you can pick the best one?

### How to use it

```bash
pip install numpy scipy matplotlib
python predict_5d.py --input data/vectors_5d.csv --horizon 5 --test-size 0.2
```

Pass `--lle 0.003132` (already the default, taken from your analysis output). You can also increase `--horizon` up to ~10–15 steps, but expect error to grow significantly past step 5 given your Lmax=5.

### What you'll get in `output/`

| File                         | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `forecast_horizon.png`       | Divergence curve, tolerance line, T_max marker |
| `predictions_comparison.png` | True vs predicted for all 3 methods × 5 dims   |
| `error_per_step.png`         | RMSE growth across the H-step horizon          |

Would you like me to retry opening the PR to commit this directly to **bisandr/Events**?
