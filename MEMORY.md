# MEMORY.md — Project Memory + Chat History

> Load this file at the start of each session to resume context.
> Update it at the end of each session with decisions, changes, and next steps.

## 1. Project Summary
- **Repo:** continual-learning-on-NoisyMNIST-data (`continual-learning-`)
- **Main entry:** `continual_learning_complete.ipynb` (12 cells, `QUICK_MODE = True` by default)
- **Scope:** NoisyMNIST data + neural model with SGD, linear IDBD teaching example. Based on Sutton & Javed Oak Lab article. No invented NetworkIDBD equations.
- **Stack:** CPU-only PyTorch, Jupyter Lab, `.venv`
- **Structure:** `src/` (data, models, optim, training, utils), `data/MNIST/`, `requirements.txt`

## 2. Key Decisions & Constraints
- 2026-09-10 — Tuning metric: rank by digit_present_clean_mse primary, clean_mse secondary, finite-first. Overall MSE is misleading (P(digit)=0.10).
- 2026-09-10 — Tuning strategy: Karpathy coarse-to-fine + successive halving (eta=3) + log-uniform random for IDBD. Fixed streams for all trials.
- 2026-09-10 — Device policy: auto cuda/cpu, same device for ExperimentConfig + ModelConfig. CPU=SGD full, IDBD skipped unless --force_idbd or GPU. IDBD ~54x slower on CPU.
- 2026-09-10 — Created tune_noisymnist_colab.py (Colab GPU + local). Validated tiny run 20/32 steps EXIT 0.
- 2026-09-10 — Divergence policy: FloatingPointError never kills sweep. run_sgd/idbd_trial returns inf-MSE finite=False diverged row, ranked last. Fixes lr=0.1 crash on T4 (commit 5b5d5e1).
- 2026-09-10 — Halving policy: report/promote best-FINITE only, keep >=2 cands until final rung. Fixes lock-in where lr=0.1 lucky at 100 steps (0.9951) then inf at 300+ stayed champion.

## 3. Current Status / Next Steps
- Benchmarked here (CPU, 41M params): SGD 0.154s/step (~2.6 min/1000), IDBD 8.378s/step (~140 min/1000), eval 0.0156s/step. Notebook comparison-cell = ~4.3h CPU (eval 1000 after every update) — must avoid.
- Next: run tune_noisymnist_colab.py --quick on Colab T4 GPU, then full --train_steps 1000 --valid_steps 1000 --idbd_configs 8. Freeze best, multi-seed confirm, CEO plots in tuning_results/.

## 4. Chat Log
### 2026-09-10 — Session 1
- Asked about last-chat history → explained no cross-session memory.
- Asked to keep chatting history → chose `MEMORY.md` (not full markdown log).
- Created this file.

### 2026-09-10 — Session 2 (hyperparameter tuning)
- Studied repo: 4096->10000->1 (~41M), tuning.py + coarse_to_fine.py exist but not wired to notebook. QUICK 8/32 steps is meaningless for tuning.
- Fetched Oak Lab article (Sutton & Javed, 2026-07-13): SGD spreads credit to all inputs, NetworkIDBD concentrates on center 28x28. Saved summary in chat.
- Measured: this machine has NO GPU (torch 2.13+cpu). Validated fix + created tune_noisymnist_colab.py.
- User on hyperparameter tuning stage, wants Colab GPU code (PyTorch) + CEO presentation.
- Auto-update policy: keep serving history without waiting.

### 2026-09-10 — Session 3 (Colab GPU run)
- User cloned repo (origin: continual-learning-on-NoisyMNIST-data.git), ready for GPU tuning.
- Gave Colab T4 steps: Runtime->GPU, pip install pandas matplotlib, --quick then full 1000/1000.
- Pending: get --quick output, then full run, freeze best, multi-seed confirm, CEO plots.
- Colab T4 --quick crashed at lr=0.1 FloatingPointError + digit_mse=1.0 for small lrs (expected: zero-init, short horizon). Fixed + pushed 5b5d5e1. Next: re-clone in Colab, rerun --quick.
- --quick rerun OK on T4: 1e-5..0.01 all digit_mse=1.0 clean=0.1211, 0.1 DIVERGED handled. No discrimination at 200 steps (ties). Explained: digit MSE != accuracy, 1.0 = predict-zero baseline (odd=1, even=-1).

### 2026-09-10 — Session 4 (digit-MSE meaning)
- Q: what is digit MSE, is it accuracy, why always 1?
- A: digit_present_clean_mse = mean (pred-clean)^2 on digit-present subset only, clean=±1. Pred-0 baseline = 1.0 exactly. Notebook QUICK also 1.0. Overall clean ~0.1 baseline (P=0.10). Need 1000-step full run for signal; 200-step ties expected.
- Next: full 1000/1000 run on T4, check mean_prediction moves off 0, digit MSE <1.0 discriminates.
- Full 1000/1000 DONE on T4 in 15.3min: coarse winner lr=0.01 (0.9995 vs 1.0), fine winner lr=0.01 (0.9995 vs 0.9999/1.0019/DIVERGED), freeze confirm 0.9962±0.0043 (seeds 11/22/33). IDBD 8cfgs all 1.0 (~105s/1000 GPU vs 5.8s SGD). Halving showed low-fidelity trap (0.1 best at 100 then inf) → fixed to best-finite + keep>=2.

### 2026-09-10 — Session 5 (full results + CEO next)
- Frozen SGD lr=0.01. Created train_final_noisymnist_colab.py: train on training pool only, fixed-validation curves every eval_every, embargoed test touched ONCE via dedicated evaluator (src evaluate_model forbids test by design), multi-seed mean±std, CEO plots + checkpoints in final_results/. Default 10k train / 1k valid / 2k test, eval_every 500.
- Validated --quick --skip_test on CPU EXIT 0 (200 steps, 42s). Next: Colab T4 --quick then full 10k x3 seeds.
- SGD FINAL DONE on T4 (10k, 69s/seed): seed11 0.8898 / test 0.8967, seed22 0.9821, seed33 0.8304 → validation 0.9008±0.0624. Real learning vs 1.0 baseline, high seed variance (only ~1000 digit hits).
- Created train_final_idbd_noisymnist_colab.py (frozen engineered IDBD meta=0.1 alpha0=1e-6 eta=0.1 tau=1e4, paired seeds/streams with SGD). 10-step CPU smoke PASS. --quick=100 steps (~100s T4). Next: IDBD 10k seed 11 (~17min) then compare.
- IDBD 10k seed 11 on T4: digit_mse stuck 1.0000 at every 500-step checkpoint (vs SGD 0.8898 same seed). Diagnosed vs Janiak/Precursor reverse-engineering note: implementation faithful (phi/meta/E/s/h all match; one noted deviation: v-floor 1e-8 vs article 1e-30, kept for float32 stability). Cause = timescale: alpha0=1e-6 needs ~18 nats growth to reach layer2 ~100, capped ≤0.1/step, digits only 10% → 10k steps insufficient. Article admits unknown Oak step count + only 1k-hidden test. Figure 2 caption: similar MSE, IDBD wins on CREDIT (0/3312 noise pixels), not MSE. Next: checkpoint credit diagnostics, then long IDBD (50k+) and/or alpha0 ablation.

### 2026-09-10 — Session 6 (IDBD diagnosis)
- IDBD 10k seed 11 flat: digit_mse 1.0000 every checkpoint, mean_pred exactly 0.0000, clean exactly 0.1000 → zero movement (alphas never grew). Mechanism: alpha0=1e-6 × tiny phi (~3e-4) → updates ~1e-10; per-step delta sign noise-dominated so meta random-walks while decay drags betas to floor.
- Pushed IDBD script v2: logs mean_alpha_w1/w2 each eval + saves optimizer_state in checkpoints + --alpha0/--meta_lr ablation flags (separate outdir). Next: credit check on old checkpoints, then alpha0=1e-4 ablation 10k seed 11.
- CREDIT RESULT (Figure 2 reproduced): SGD seed11 → 3312/3312 noise pixels above 0.0002, center 597, max|w|=0.078 (credit everywhere). IDBD seed11 → 0/3312 noise, 0 center, max|w|=1e-4 (asleep, clean). Ablation alpha0=1e-4 running: a1/a2 sinking (9.98e-5→9.93e-5), decay winning so far, digit_mse 1.0 @2500. Await 10k.

<!-- Append new sessions below as ### YYYY-MM-DD — Session N -->
