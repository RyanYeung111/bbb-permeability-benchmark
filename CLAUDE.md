# CLAUDE.md

This file provides guidance for working in this repository.

## Environment

- This project uses the conda environment **bbb**, located at `C:\Users\gigar\miniforge3\envs\bbb`.
- Python version: **3.11**.
- **All Python commands must use that interpreter** — do not use a system/global Python or any other environment.
  - Interpreter path: `C:\Users\gigar\miniforge3\envs\bbb\python.exe`
  - Example: `C:\Users\gigar\miniforge3\envs\bbb\python.exe script.py`
  - Or activate first: `conda activate bbb`

## Dependency notes

- **PyTDC** is pinned to version **0.4.1** and must be installed with **`--no-deps`**.
- **setuptools** is pinned to **below version 81** (`setuptools<81`).
