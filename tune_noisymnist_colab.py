"""NoisyMNIST hyperparameter tuning — runs locally (CPU) AND in Google Colab (GPU).

Colab quickstart:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload this repo (or git clone it), then:
     !pip install -q pandas matplotlib  # torch/torchvision already in Colab
  3. Run: !python tune_noisymnist_colab.py --quick
     Full: !python tune_noisymnist_colab.py --train_steps 1000 --valid_steps 1000 --idbd_configs 8

Local:
  source .venv/bin/activate
  python tune_noisymnist_colab.py --quick
  python tune_noisymnist_colab.py --train_steps 1000 --valid_steps 1000

Strategy (Karpathy coarse-to-fine + successive halving + random search):
  1. COARSE cheap fidelity: wide log grid on short fixed streams, same experiences
     for every trial. Rank by digit_present_clean_mse -> clean_mse -> finite.
  2. ZOOM: interval between best neighbours, 5 dense log points (Karpathy).
  3. HALVING: train-steps as resource (100->300->1000, keep top 1/3). ~70% saving.
  4. IDBD: log-uniform random (meta_lr, alpha0, eta, tau) — grid is wasteful in 4-D.
  5. FREEZE + multi-seed confirm on FRESH streams, report mean +/- std.

Why digit-present MSE is primary: P(digit)=0.10, so always-predict-0 looks good
on overall MSE but learns nothing. ~100 digits per 1000 steps; 8 steps ~= 0-1
digits, so QUICK results are illustrative only.
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
    make_fixed_stream,
    make_train_validation_partitions,
)
from src.models import ModelConfig
from src.training.coarse_to_fine import (
    karpathy_zoom,
    rank_trials,
    sample_network_idbd_configs,
    sgd_coarse_grid,
    successive_halving_schedule,
)
from src.training.tuning import run_network_idbd_trial, run_sgd_trial, SGDTrialConfig
from src.utils import seed_everything


def get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev} | torch {torch.__version__}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("CPU mode: SGD is fine (~2.6 min/1000 steps). "
              "IDBD is ~54x slower on CPU (~140 min/1000 steps) — use Colab GPU.")
    return dev


def build_streams(data_root: str, device: str, train_steps: int, valid_steps: int,
                  train_seed=314159, valid_seed=271828):
    cfg = ExperimentConfig(device=device)
    parts = make_train_validation_partitions(data_root, cfg, download=(data_root.startswith("/content")))
    train_exps = list(make_fixed_stream(parts.training_pool, train_steps, train_seed, cfg))
    valid_exps = list(make_fixed_stream(parts.validation_pool, valid_steps, valid_seed, cfg))
    n_digits = sum(1 for e in valid_exps if e["digit_present"])
    print(f"train={train_steps} valid={valid_steps} valid_digits={n_digits} "
          f"(~{valid_steps*0.1:.0f} expected)")
    if n_digits == 0:
        print("WARNING: 0 digits in validation — digit_present_mse will be NaN. Increase valid_steps.")
    return cfg, train_exps, valid_exps


def tune_sgd(mconfig: ModelConfig, train_exps, valid_exps, init_seed: int,
             coarse: list[float] | None = None):
    coarse = coarse or sgd_coarse_grid()
    print(f"\n=== SGD COARSE {coarse} ===")
    rows = []
    for lr in coarse:
        cfg_t = SGDTrialConfig(lr, init_seed, 314159, 271828, len(train_exps), len(valid_exps))
        t0 = time.time()
        try:
            r = run_sgd_trial(mconfig, train_exps, valid_exps, cfg_t)
        except FloatingPointError as e:
            # Belt-and-suspenders: run_sgd_trial already returns a diverged row,
            # but never let one lr kill the whole sweep.
            r = {"learning_rate": lr, "digit_present_clean_target_mse": float("inf"),
                 "clean_target_mse": float("inf"), "finite": False, "diverge_reason": str(e)}
            print(f"  lr={lr:<8} DIVERGED: {e}")
        else:
            status = "DIVERGED" if not r["finite"] else f"digit_mse={r['digit_present_clean_target_mse']:.4f}"
            print(f"  lr={lr:<8} {status} "
                  f"clean={r['clean_target_mse']:.4f} finite={r['finite']} ({time.time()-t0:.1f}s)")
        rows.append(r)
    coarse_df = rank_trials(pd.DataFrame(rows))
    best = float(coarse_df.iloc[0]["learning_rate"])
    print(f"coarse winner: lr={best}")

    fine_grid = karpathy_zoom(best, coarse)
    print(f"=== SGD FINE (Karpathy zoom) {['%.4g' % v for v in fine_grid]} ===")
    fine_rows = []
    for lr in fine_grid:
        cfg_t = SGDTrialConfig(float(lr), init_seed, 314159, 271828, len(train_exps), len(valid_exps))
        try:
            r = run_sgd_trial(mconfig, train_exps, valid_exps, cfg_t)
        except FloatingPointError as e:
            r = {"learning_rate": float(lr), "digit_present_clean_target_mse": float("inf"),
                 "clean_target_mse": float("inf"), "finite": False, "diverge_reason": str(e)}
            print(f"  lr={lr:<8.5f} DIVERGED: {e}")
        else:
            status = "DIVERGED" if not r["finite"] else f"digit_mse={r['digit_present_clean_target_mse']:.4f}"
            print(f"  lr={lr:<8.5f} {status}")
        fine_rows.append(r)
    fine_df = rank_trials(pd.DataFrame(fine_rows))
    print(f"fine winner: lr={float(fine_df.iloc[0]['learning_rate'])}")
    return coarse_df, fine_df


def tune_sgd_halving(mconfig: ModelConfig, full_train, valid_exps, candidates, init_seed: int):
    rungs = successive_halving_schedule(100 if len(full_train) >= 300 else max(10, len(full_train)//4),
                                        len(full_train), eta=3)
    print(f"\n=== SUCCESSIVE HALVING rungs={rungs} ===")
    per_rung, cands = {}, [float(v) for v in candidates]
    for ri, rung in enumerate(rungs):
        prefix = full_train[:rung]
        rows = []
        for lr in cands:
            try:
                rows.append(run_sgd_trial(mconfig, prefix, valid_exps,
                    SGDTrialConfig(lr, init_seed, 314159, 271828, rung, len(valid_exps))))
            except FloatingPointError as e:
                rows.append({"learning_rate": lr, "digit_present_clean_target_mse": float("inf"),
                             "clean_target_mse": float("inf"), "finite": False, "diverge_reason": str(e)})
        df = rank_trials(pd.DataFrame(rows))
        per_rung[rung] = df
        finite_df = df[df["finite"].astype(bool)]
        # Report best-finite (a diverged low-fidelity lucky winner must not mask the signal).
        rep = finite_df.iloc[0] if len(finite_df) else df.iloc[0]
        print(f"  rung {rung}: best lr={float(rep['learning_rate'])} "
              f"digit_mse={float(rep['digit_present_clean_target_mse']):.4f} "
              f"finite={bool(rep['finite'])} (tested {len(cands)})")
        # Keep top 1/3 of FINITE configs when any exist; keep >=2 until final rung
        # so one lucky low-fidelity rung cannot lock in a later-diverging lr (e.g. 0.1).
        pool = finite_df if len(finite_df) else df
        last = (ri == len(rungs) - 1)
        keep = max(1 if last else min(2, len(pool)), len(cands) // 3)
        cands = [float(v) for v in pool.iloc[:keep]["learning_rate"].tolist()]
    return per_rung


def tune_idbd(mconfig: ModelConfig, train_exps, valid_exps, n_configs: int, rng_seed: int):
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    cfgs = sample_network_idbd_configs(n_configs, rng, 20260202, 314159, 271828,
                                       len(train_exps), len(valid_exps))
    print(f"\n=== NetworkIDBD random n={n_configs} (log-uniform) ===")
    rows = []
    for i, c in enumerate(cfgs):
        t0 = time.time()
        try:
            r = run_network_idbd_trial(mconfig, train_exps, valid_exps, c)
            dt = time.time() - t0
            print(f"  [{i+1}/{n_configs}] meta={c.meta_lr:.4f} alpha0={math.exp(c.initial_beta):.1e} "
                  f"eta={c.eta:.3f} tau={c.tau:.0f} -> digit_mse={r['digit_present_clean_target_mse']:.4f} ({dt:.1f}s)")
            rows.append(r)
        except FloatingPointError as e:
            print(f"  [{i+1}/{n_configs}] DIVERGED: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["alpha0"] = df["initial_beta"].apply(lambda b: math.exp(float(b)))
    return rank_trials(df)


def confirm_multiseed(mconfig: ModelConfig, data_root: str, device: str, best_lr: float,
                      train_steps: int, valid_steps: int, seeds=(11, 22, 33)):
    """Freeze best lr, evaluate on FRESH streams with new init+stream seeds."""
    print(f"\n=== FREEZE + confirm lr={best_lr} on {len(seeds)} fresh seeds ===")
    import numpy as np
    mses = []
    for s in seeds:
        seed_everything(1000 + s)
        cfg = ExperimentConfig(device=device)
        parts = make_train_validation_partitions(data_root, cfg)
        tr = list(make_fixed_stream(parts.training_pool, train_steps, 314159 + s, cfg))
        va = list(make_fixed_stream(parts.validation_pool, valid_steps, 271828 + s, cfg))
        r = run_sgd_trial(mconfig, tr, va, SGDTrialConfig(best_lr, 20260202 + s, 314159 + s, 271828 + s, train_steps, valid_steps))
        mses.append(float(r["digit_present_clean_target_mse"]))
        print(f"  seed {s}: digit_mse={mses[-1]:.4f}")
    arr = np.array(mses)
    print(f"FINAL: {arr.mean():.4f} +/- {arr.std():.4f} (n={len(arr)})")
    return mses


def ceo_plots(outdir: Path, coarse_df, fine_df, idbd_df):
    outdir.mkdir(parents=True, exist_ok=True)
    # 1. SGD tuning curve (log-x) — the CEO slide
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for df, label, marker in [(coarse_df, "coarse", "o"), (fine_df, "fine (zoom)", "s")]:
        ax.semilogx(df["learning_rate"], df["digit_present_clean_target_mse"], marker + "-", label=label)
    ax.set_xlabel("SGD learning rate (log scale)")
    ax.set_ylabel("digit-present clean MSE (lower = better)")
    ax.set_title("NoisyMNIST tuning: coarse -> Karpathy zoom")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "sgd_tuning_curve.png", dpi=150)
    # 2. IDBD random results
    if idbd_df is not None and len(idbd_df):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(range(len(idbd_df)), idbd_df["digit_present_clean_target_mse"])
        ax.set_xlabel("IDBD config rank"); ax.set_ylabel("digit-present clean MSE")
        ax.set_title("NetworkIDBD random search (ranked)")
        fig.tight_layout(); fig.savefig(outdir / "idbd_ranked.png", dpi=150)
    print(f"plots -> {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="200 train / 256 valid demo")
    ap.add_argument("--train_steps", type=int, default=1000)
    ap.add_argument("--valid_steps", type=int, default=1000)
    ap.add_argument("--data_root", type=str, default="data/MNIST")
    ap.add_argument("--outdir", type=str, default="tuning_results")
    ap.add_argument("--idbd_configs", type=int, default=8)
    ap.add_argument("--idbd_steps", type=int, default=0, help="0=auto (full on GPU, 50 on CPU)")
    ap.add_argument("--force_idbd", action="store_true", help="run IDBD even on CPU")
    ap.add_argument("--skip_idbd", action="store_true")
    ap.add_argument("--init_seed", type=int, default=20260202)
    args = ap.parse_args()

    if args.quick:
        args.train_steps, args.valid_steps, args.idbd_configs = 200, 256, 3

    device = get_device()
    mconfig = ModelConfig(device=device.type)
    _, train_exps, valid_exps = build_streams(args.data_root, device.type, args.train_steps, args.valid_steps)

    # Guard: IDBD on CPU is ~8.4s/step (measured here, 41M params) vs SGD ~0.15s/step.
    # Full 1000-step IDBD = ~140 min/trial on CPU. On Colab T4 GPU it is ~10-20x faster.
    # Auto policy: GPU -> full horizon. CPU -> 50-step smoke unless --force_idbd.
    if args.skip_idbd:
        idbd_train, idbd_n, run_idbd = [], 0, False
    elif device.type == "cpu" and not args.force_idbd:
        print("CPU auto-policy: SKIPPING IDBD search (would be ~hours). SGD runs full horizon. "
              "Use Colab GPU or --force_idbd to run IDBD.")
        idbd_train, idbd_n, run_idbd = [], 0, False
    else:
        auto_steps = len(train_exps) if device.type == "cuda" else 50
        use_steps = args.idbd_steps if args.idbd_steps > 0 else auto_steps
        idbd_train, idbd_n, run_idbd = train_exps[:use_steps], args.idbd_configs, True
        print(f"IDBD search horizon: {len(idbd_train)} steps x {idbd_n} configs on {device.type}")

    t_all = time.time()
    coarse_df, fine_df = tune_sgd(mconfig, train_exps, valid_exps, args.init_seed)
    _ = tune_sgd_halving(mconfig, train_exps, valid_exps,
                         coarse_df["learning_rate"].tolist(), args.init_seed)

    idbd_df = pd.DataFrame()
    if run_idbd:
        idbd_df = tune_idbd(mconfig, idbd_train, valid_exps, idbd_n, rng_seed=0)

    best_lr = float(fine_df.iloc[0]["learning_rate"])
    if not args.quick:
        confirm_multiseed(mconfig, args.data_root, device.type, best_lr, args.train_steps, args.valid_steps)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    coarse_df.to_csv(outdir / "sgd_coarse.csv", index=False)
    fine_df.to_csv(outdir / "sgd_fine.csv", index=False)
    if len(idbd_df):
        idbd_df.to_csv(outdir / "idbd_random.csv", index=False)
    ceo_plots(outdir, coarse_df, fine_df, idbd_df)
    print(f"\nDONE in {(time.time()-t_all)/60:.1f} min. Best SGD lr={best_lr}. CSVs+PNGs in {outdir}/")


if __name__ == "__main__":
    main()
