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

<!-- Append new sessions below as ### YYYY-MM-DD — Session N -->
