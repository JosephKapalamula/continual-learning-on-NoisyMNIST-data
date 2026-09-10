"""Final frozen NetworkIDBD for NoisyMNIST — paired comparison with SGD final.

Frozen config = notebook engineered defaults (inside tuned random range):
  meta_lr=0.1, alpha0=1e-6, decay=0.9995, beta in [1e-8, 1e4], eta=0.1, tau=1e4.
Tuning 8 random configs all tied at 1.0 @1000 steps, so no empirical winner to
freeze; this default is the honest prior. Report as engineered approximation
(Oak Lab never published NetworkIDBD equations).

Paired protocol (matches train_final_noisymnist_colab.py):
  same seeds -> same streams as SGD (314159+seed train, 271828+seed valid),
  same eval_every, same fixed validation, test touched ONCE at end.
Compare final_results/ (SGD) vs final_idbd_results/ (IDBD) on same seeds.

Cost on T4 GPU: ~105s/1000 steps (~17min/10k per seed). Start with 1 seed:
  !python train_final_idbd_noisymnist_colab.py --quick
  !python train_final_idbd_noisymnist_colab.py --train_steps 10000 --eval_every 500 --seeds 11
Then optionally --seeds 11 22 33 (~52min total; run overnight / background).

Outputs in final_idbd_results/: learning_curves.csv, final_validation.csv,
final_test.csv (once), learning_curve.png, checkpoints/seed_*.pt
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
from src.optim import NetworkIDBD
from src.training import (
    LossConvention,
    evaluate_model,
    online_network_idbd_update,
)
from src.utils import seed_everything


# FROZEN engineered defaults (notebook comparison cell + tuned range center)
FROZEN = dict(
    meta_lr=0.1,
    initial_beta=math.log(1e-6),
    decay=0.9995,
    beta_min=math.log(1e-8),
    beta_max=math.log(1e4),
    eta=0.1,
    tau=1e4,
)


def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev} | torch {torch.__version__}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FROZEN IDBD: meta={FROZEN['meta_lr']} alpha0={math.exp(FROZEN['initial_beta']):.0e} "
          f"eta={FROZEN['eta']} tau={FROZEN['tau']:.0f} decay={FROZEN['decay']}")
    return dev


def evaluate_on_test_pool(model, test_pool, num_steps: int, seed: int, config) -> dict:
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
                assert exp["source_partition"] == "official_test"
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


def train_one_seed(seed, data_root, device, train_steps, valid_steps, eval_every, outdir) -> dict:
    seed_everything(1000 + seed)
    cfg = ExperimentConfig(device=device.type)
    parts = make_train_validation_partitions(data_root, cfg)
    mconfig = ModelConfig(device=device.type)
    seed_everything(20260202 + seed)
    model = NoisyMNISTMLP(mconfig)
    opt = NetworkIDBD(model.parameters(), **FROZEN)

    valid_exps = list(make_fixed_stream(parts.validation_pool, valid_steps, 271828 + seed, cfg))
    train_stream = NoisyMNISTStream(parts.training_pool, cfg, seed=314159 + seed)

    curve = []
    t0 = time.time()
    for step in range(1, train_steps + 1):
        exp = next(train_stream)
        online_network_idbd_update(
            model, opt, exp,
            LossConvention.MEAN_SQUARED_ERROR,
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
            print(f"    seed {seed} step {step}: digit_mse={curve[-1]['digit_present_clean_target_mse']:.4f}",
                  flush=True)
    dt = time.time() - t0
    final = curve[-1]
    print(f"  seed {seed}: step {train_steps} digit_mse={final['digit_present_clean_target_mse']:.4f} "
          f"clean={final['clean_target_mse']:.4f} mean_pred={final['mean_prediction']:.4f} ({dt:.0f}s)")

    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"seed": seed, "frozen": FROZEN, "train_steps": train_steps,
                "state_dict": model.state_dict()},
               outdir / "checkpoints" / f"seed_{seed}.pt")
    return {"curve": curve, "model": model, "config": cfg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train_steps", type=int, default=10000)
    ap.add_argument("--valid_steps", type=int, default=1000)
    ap.add_argument("--test_steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11])
    ap.add_argument("--data_root", type=str, default="data/MNIST")
    ap.add_argument("--outdir", type=str, default="final_idbd_results")
    ap.add_argument("--skip_test", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.train_steps, args.valid_steps, args.test_steps = 100, 128, 128
        args.eval_every, args.seeds = 50, [11]

    print(f"IDBD FINAL | train={args.train_steps} valid={args.valid_steps} "
          f"test={args.test_steps} eval_every={args.eval_every} seeds={args.seeds}")
    device = get_device()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_curves, finals, test_rows = [], [], []
    for seed in args.seeds:
        res = train_one_seed(seed, args.data_root, device, args.train_steps,
                             args.valid_steps, args.eval_every, outdir)
        all_curves.extend(res["curve"])
        finals.append({"seed": seed, **res["curve"][-1]})
        if not args.skip_test and seed == args.seeds[0]:
            test_pool = make_test_pool(args.data_root, res["config"])
            tm = evaluate_on_test_pool(res["model"], test_pool, args.test_steps,
                                       seed=161803 + seed, config=res["config"])
            test_rows.append({"seed": seed, **tm})
            print(f"  TEST-ONCE seed {seed}: digit_mse={tm['digit_present_clean_target_mse']:.4f} "
                  f"clean={tm['clean_target_mse']:.4f} n_digits={tm['digit_present_count']}")

    curve_df = pd.DataFrame(all_curves)
    final_df = pd.DataFrame(finals)
    curve_df.to_csv(outdir / "learning_curves.csv", index=False)
    final_df.to_csv(outdir / "final_validation.csv", index=False)
    if test_rows:
        pd.DataFrame(test_rows).to_csv(outdir / "final_test.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed, g in curve_df.groupby("seed"):
        ax.plot(g["step"], g["digit_present_clean_target_mse"], marker="o", ms=3, label=f"IDBD seed {seed}")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="predict-zero baseline")
    ax.set_xlabel("online training steps (frozen IDBD)")
    ax.set_ylabel("digit-present clean MSE (lower = better)")
    ax.set_title("Final training: NetworkIDBD learning curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "learning_curve.png", dpi=150)

    vals = final_df["digit_present_clean_target_mse"].to_numpy()
    print(f"\nFINAL IDBD validation digit_mse: {vals.mean():.4f} ± {vals.std():.4f} (n={len(vals)})")
    print(f"DONE. CSVs+PNGs+checkpoints in {outdir}/")


if __name__ == "__main__":
    main()
