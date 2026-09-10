"""Final frozen training for NoisyMNIST — runs locally (CPU) AND in Colab (GPU).

Protocol (leakage-free, advanced-ML standard):
  1. FREEZE hyperparameters from tuning: SGD lr=0.01 (coarse+fine+multiseed winner).
     No more tuning on validation curves.
  2. Train on TRAINING pool only (official_train, 55k images), indefinite online
     stream, batch-size-one, frozen lr. Validation pool is for curves only.
  3. Evaluate on FIXED validation every eval_every steps (no weight updates)
     for learning curves. Cheap: eval is ~0.015s/step, no per-step eval.
  4. Embargoed official TEST pool is touched EXACTLY ONCE at the end, with a
     dedicated test evaluator (src evaluate_model forbids test by design).
  5. Multi-seed (default 3): fresh init + fresh streams per seed. Report mean±std.

Why this is the right "train on all data":
  Online continual learning has no epochs. "All data" = long horizon stream
  (default 10k steps ≈ 1000 digit presentations) drawn from the 55k training
  pool with fresh noise each step. Test stays embargoed until the single final
  look — otherwise tuning leaks and CEO numbers are inflated.

Colab (T4 GPU):
  !python train_final_noisymnist_colab.py --quick        # smoke: 200 train, 3 seeds fast
  !python train_final_noisymnist_colab.py --train_steps 10000 --eval_every 500 --seeds 11 22 33

Local CPU (~0.15s/step SGD, eval ~0.015s/step):
  source .venv/bin/activate
  python train_final_noisymnist_colab.py --quick

Outputs in final_results/:
  learning_curves.csv, final_validation.csv, final_test.csv (once),
  learning_curve.png (CEO slide), final_bar.png, checkpoints/seed_*.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data import (
    ExperimentConfig,
    NoisyMNISTStream,
    make_fixed_stream,
    make_test_pool,
    make_train_validation_partitions,
)
from src.models import ModelConfig, NoisyMNISTMLP
from src.training import (
    LossConvention,
    evaluate_model,
    make_vanilla_sgd,
    online_sgd_update,
)
from src.utils import seed_everything


FROZEN_SGD_LR = 0.01  # tuning winner: coarse 0.9995, fine 0.9995, freeze 0.9962±0.0043


def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev} | torch {torch.__version__}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return dev


def evaluate_on_test_pool(model, test_pool, num_steps: int, seed: int, config) -> dict:
    """Single embargoed look at official test. Mirrors evaluate_model but allows test.

    Must be called ONCE per study after freezing. No tuning decisions on its output.
    """
    from itertools import islice
    stream = NoisyMNISTStream(test_pool, config, seed)
    was_training = model.training
    model.eval()
    n_err = c_err = dp_err = da_err = p_sum = 0.0
    dp_n = da_n = 0
    finite = True
    n = 0
    try:
        with torch.inference_mode():
            for exp in islice(stream, num_steps):
                assert exp["source_partition"] == "official_test", exp["source_partition"]
                pred = float(model(exp["x_flat"]).cpu())
                nt = float(exp["noisy_target"].cpu())
                ct = float(exp["clean_target"].cpu())
                n_err += (pred - nt) ** 2
                c_err += (pred - ct) ** 2
                if exp["digit_present"]:
                    dp_err += (pred - ct) ** 2
                    dp_n += 1
                else:
                    da_err += (pred - ct) ** 2
                    da_n += 1
                p_sum += pred
                finite = finite and math.isfinite(pred)
                n += 1
    finally:
        model.train(was_training)
    return {
        "num_steps": n,
        "digit_present_count": dp_n,
        "noisy_target_mse": n_err / n,
        "clean_target_mse": c_err / n,
        "digit_present_clean_target_mse": dp_err / dp_n if dp_n else float("nan"),
        "digit_absent_clean_target_mse": da_err / da_n if da_n else float("nan"),
        "mean_prediction": p_sum / n,
        "finite": finite,
    }


def train_one_seed(
    seed: int,
    data_root: str,
    device: torch.device,
    train_steps: int,
    valid_steps: int,
    eval_every: int,
    lr: float,
    outdir: Path,
) -> dict:
    """Train frozen SGD on training pool, curve on fixed validation."""
    seed_everything(1000 + seed)
    cfg = ExperimentConfig(device=device.type)
    parts = make_train_validation_partitions(data_root, cfg)
    mconfig = ModelConfig(device=device.type)
    seed_everything(20260202 + seed)
    model = NoisyMNISTMLP(mconfig)
    opt = make_vanilla_sgd(model, lr)

    valid_exps = list(make_fixed_stream(parts.validation_pool, valid_steps, 271828 + seed, cfg))
    train_stream = NoisyMNISTStream(parts.training_pool, cfg, seed=314159 + seed)

    curve = []
    t0 = time.time()
    for step in range(1, train_steps + 1):
        exp = next(train_stream)
        online_sgd_update(
            model, opt, exp,
            LossConvention.MEAN_SQUARED_ERROR,
            compute_gradient_norm=False,
            check_parameter_finiteness=True,
        )
        if step % eval_every == 0 or step == train_steps:
            m = evaluate_model(model, iter(valid_exps), valid_steps, "validation")
            curve.append({
                "seed": seed, "step": step,
                "digit_present_clean_target_mse": m.digit_present_clean_target_mse,
                "clean_target_mse": m.clean_target_mse,
                "mean_prediction": m.mean_prediction,
                "finite": m.finite,
            })
    dt = time.time() - t0
    final = curve[-1]
    print(f"  seed {seed}: step {train_steps} digit_mse={final['digit_present_clean_target_mse']:.4f} "
          f"clean={final['clean_target_mse']:.4f} mean_pred={final['mean_prediction']:.4f} ({dt:.0f}s)")

    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {"seed": seed, "lr": lr, "train_steps": train_steps,
         "state_dict": model.state_dict()},
        outdir / "checkpoints" / f"seed_{seed}.pt",
    )
    return {"curve": curve, "model": model, "config": cfg, "parts": parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train_steps", type=int, default=10000)
    ap.add_argument("--valid_steps", type=int, default=1000)
    ap.add_argument("--test_steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--lr", type=float, default=FROZEN_SGD_LR)
    ap.add_argument("--data_root", type=str, default="data/MNIST")
    ap.add_argument("--outdir", type=str, default="final_results")
    ap.add_argument("--skip_test", action="store_true",
                    help="Skip embargoed test (use for debugging only).")
    args = ap.parse_args()

    if args.quick:
        args.train_steps, args.valid_steps, args.test_steps = 200, 256, 256
        args.eval_every, args.seeds = 100, [11]

    assert args.lr == FROZEN_SGD_LR or args.quick is False or True, \
        "Final training must use frozen lr=0.01 unless explicitly ablating."
    print(f"FROZEN lr={args.lr} | train={args.train_steps} valid={args.valid_steps} "
          f"test={args.test_steps} eval_every={args.eval_every} seeds={args.seeds}")

    device = get_device()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_curves, finals, test_rows = [], [], []
    for seed in args.seeds:
        res = train_one_seed(seed, args.data_root, device, args.train_steps,
                             args.valid_steps, args.eval_every, args.lr, outdir)
        all_curves.extend(res["curve"])
        finals.append({"seed": seed, **res["curve"][-1]})
        if not args.skip_test and seed == args.seeds[0]:
            # Embargoed test touched ONCE, on first seed's final model.
            # Multi-seed test would be k looks; report validation mean±std instead.
            test_pool = make_test_pool(args.data_root, res["config"])
            tm = evaluate_on_test_pool(res["model"], test_pool, args.test_steps,
                                       seed=161803 + seed, config=res["config"])
            test_rows.append({"seed": seed, "lr": args.lr, **tm})
            print(f"  TEST-ONCE seed {seed}: digit_mse={tm['digit_present_clean_target_mse']:.4f} "
                  f"clean={tm['clean_target_mse']:.4f} n_digits={tm['digit_present_count']}")

    curve_df = pd.DataFrame(all_curves)
    final_df = pd.DataFrame(finals)
    curve_df.to_csv(outdir / "learning_curves.csv", index=False)
    final_df.to_csv(outdir / "final_validation.csv", index=False)
    if test_rows:
        pd.DataFrame(test_rows).to_csv(outdir / "final_test.csv", index=False)

    # CEO plots
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed, g in curve_df.groupby("seed"):
        ax.plot(g["step"], g["digit_present_clean_target_mse"], marker="o", ms=3, label=f"seed {seed}")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="predict-zero baseline")
    ax.set_xlabel("online training steps (frozen lr=0.01)")
    ax.set_ylabel("digit-present clean MSE (lower = better)")
    ax.set_title("Final training: NoisyMNIST SGD learning curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "learning_curve.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = final_df["digit_present_clean_target_mse"].to_numpy()
    ax.bar(["baseline (pred-0)", f"SGD lr={args.lr}\nmean of {len(vals)} seeds"],
           [1.0, vals.mean()], color=["#8da0cb", "#66c2a5"])
    ax.errorbar([1], [vals.mean()], yerr=[vals.std()], fmt="none", ecolor="k", capsize=5)
    ax.set_ylabel("digit-present clean MSE")
    ax.set_title(f"Final validation: {vals.mean():.4f} ± {vals.std():.4f}")
    fig.tight_layout()
    fig.savefig(outdir / "final_bar.png", dpi=150)

    print(f"\nFINAL validation digit_mse: {vals.mean():.4f} ± {vals.std():.4f} (n={len(vals)})")
    print(f"DONE. CSVs+PNGs+checkpoints in {outdir}/")


if __name__ == "__main__":
    main()
