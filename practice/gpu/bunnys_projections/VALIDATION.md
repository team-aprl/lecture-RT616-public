# Validation

Initial public package checked on 2026-09-09.

## Passed locally

- Fresh Python 3.12.14 environment created by setup.py and all pinned direct dependencies installed.
- Windows long-path installation issue reproduced, then avoided using the short per-project environment path under LOCALAPPDATA.
- NVIDIA GeForce RTX 5070 Ti Laptop GPU; CuPy 14.2.0; CUDA runtime 12.9; driver API 13.1.
- doctor.py: actual CUDA compilation, analytic projection, near-plane rejection, z-buffer occlusion, and CPU/GPU array equality.
- Explicit CPU-only mode, including a three-frame console benchmark and JSON output.
- run.ps1 launched the actual installed environment on an alternate port.
- HTTP K batch with 64×64 Bunny, four matching camera poses: four records returned, completion statistics present, first/last CPU/GPU depth checks passed.
- Browser JavaScript syntax check; PNG frame endpoint and streaming batch log endpoint.
- Public standalone notebook: noninteractive setup and console cells executed without errors.
- Models and five trajectories were previously checked in the source exercise. No model archives are part of this package.

The 64×64 check above is a functional test, not a statistically meaningful performance claim. Use K=60 or more on your own device for comparisons.

## Automated coverage

The GitHub Actions workflow installs the CPU dependencies on Windows and Ubuntu, then runs the offline analytic projection test. It does not need Stanford downloads. The run status on GitHub is the authority for these hosted checks.

## Not verified

- Linux **GPU** execution, AMD/Intel/Apple GPU support (not implemented), and hosted Colab GPU execution.
- Docker: not provided or needed for the native workflow.
- Other NVIDIA laptop models and driver versions. The setup script performs a real kernel check so incompatible environments fail before opening the viewer.
