# BBB Permeability Prediction

Predicting blood-brain barrier (BBB) permeability of small molecules using the
`BBB_Martins` dataset from Therapeutics Data Commons (TDC).

## Approach

- **Dataset:** [TDC](https://tdcommons.ai/)'s `BBB_Martins` benchmark
  (binary BBB permeability labels).
- **Split:** Scaffold split, so structurally distinct molecule scaffolds are
  held out for evaluation rather than randomly split.
- **Features:** Morgan (circular) fingerprints computed from molecular
  structure.
- **Model:** Random forest classifier trained on the fingerprint features.

## Baseline results

| Metric | Value |
| --- | --- |
| AUROC | 0.907 ± 0.017 |

Reported as mean ± standard deviation across five random seeds.

## Environment

This project uses the conda environment `bbb` (Python 3.11) at
`C:\Users\gigar\miniforge3\envs\bbb`. All Python commands should be run with
that interpreter — see [CLAUDE.md](CLAUDE.md) for details, including the
pinned `PyTDC==0.4.1` (installed with `--no-deps`) and `setuptools<81`
dependency notes.

## Project structure

- [bbb_step1.py](bbb_step1.py) — data loading, scaffold split, fingerprint
  featurization, and baseline random forest training/evaluation.
- `data/` — local dataset cache (not tracked in git).

## Reproducing
 
```bash
conda env create -f environment.yml
conda activate bbb
```
 
If the pip step fails on PyTDC, install it separately — later versions of its dependency tree include a package with no Windows distribution:
 
```bash
pip install PyTDC==0.4.1 --no-deps
pip install fuzzywuzzy tqdm seaborn requests huggingface_hub
```
 
On Windows 11 with Smart App Control enabled, pip-installed packages with compiled extensions may be blocked at import. Installing them from conda-forge resolves this.