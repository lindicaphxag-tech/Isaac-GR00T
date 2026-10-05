# CASJ public PushT validation harness

This branch is a **validation surface only**. It is not a proposed NVIDIA
Isaac-GR00T change and does not imply NVIDIA adoption of CASJ.

The workflow pins:
- LeRobot commit: 8c920c4270460851cedd2737657584586d3dc66f
- CASJ paper-core commit: f11893787c38609032a2756c56f95e97730225bf
- public checkpoint: lerobot/diffusion_pusht (selected by the CASJ script)
- public dataset metadata: lerobot/pusht

The first run is intentionally a CPU smoke gate with 10 reverse diffusion
steps and one exact-state snapshot. The JSON artifact records both the
checkpoint-default and effective inference-step counts, so this run must not be
reported as default-checkpoint performance.

If the smoke gate shows interpretable physical derivatives and completes
cleanly, a later full gate should use the checkpoint-default inference steps and
multiple snapshots.
