# Continual Learning with NoisyMNIST

This repository contains one main, presentation-friendly notebook:

**`continual_learning_complete.ipynb`**

It explains the project from data and batch-size-one online learning through the
model architecture, evaluation, SGD training, linear IDBD, results, plots, and checks.

## Scientific scope

The NoisyMNIST data design and neural architecture come from Sutton and Javed's Oak
Lab article, *Learning from experience instead of curated datasets*. The notebook
implements the neural model with SGD and demonstrates canonical IDBD separately on a
linear problem. It does not claim that these are a final fair neural comparison, and
it does not invent Oak Lab's unpublished NetworkIDBD equations.

## 1. Clone the repository

```bash
git clone https://github.com/JosephKapalamula/continual-learning-.git
cd continual-learning-
```

The trailing hyphen in `continual-learning-` is part of the repository name.

## 2. Create and activate an environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If Debian or Ubuntu reports that `ensurepip` is unavailable, run:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

Then repeat the environment creation and activation commands.

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements use CPU-only PyTorch, so an NVIDIA GPU is not required. Installation
uses approximately 1.3 GB and may take several minutes.

## 4. Run the main notebook

```bash
jupyter lab continual_learning_complete.ipynb
```

Select the Python kernel from `.venv` if prompted, then choose **Run → Run All Cells**.
The notebook defaults to `QUICK_MODE = True`, which is a short educational run rather
than a long final-performance experiment.

## Expected output

All 12 code cells should run without errors and show:

- NoisyMNIST examples and data configuration;
- shapes, normalization, and online batching;
- architecture and parameter count;
- SGD training and fixed-validation results;
- a linear SGD-versus-IDBD teaching example;
- result plots and reproducibility checks marked `PASS`.

The official MNIST test partition is intentionally not used in this development run.

## Project structure

```text
continual-learning-/
├── continual_learning_complete.ipynb  # Start here
├── requirements.txt                   # Python dependencies
├── src/                               # Data, model, optimizer, and training code
└── data/MNIST/                        # MNIST files used by the notebook
```

## Reopen the project later

```bash
cd continual-learning-
source .venv/bin/activate
jupyter lab continual_learning_complete.ipynb
```

Exit the environment with `deactivate`.
