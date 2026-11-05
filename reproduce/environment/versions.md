# The environment behind the official evaluations

All three evaluated submissions (`student_v1`, `student_v1_2`, `student_v1_3`) were served from one Python environment, on one
machine. No package in it changed after 2026-10-20, before the first official evaluation.

## Serving

Serving needs only the packages in [`requirements-inference.txt`](requirements-inference.txt): the eight the server imports, plus pillow for
torchvision, and their dependencies.
Isaac Sim, Isaac Lab and the challenge kit are not imported: the kit's `tower_protocol.py` is vendored in
`tower_rl/server/`, and the only file needed from the challenge site is `track.json`.

```bash
python3.11 -m venv venv && venv/bin/pip install -r reproduce/environment/requirements-inference.txt
```

With `uv`, add `--index-strategy unsafe-best-match`. Checked on 2026-11-05: every `eval/*` tag, served from a clean clone in such
an environment, passes `scripts/probe_server.py` both with blank images and with organizer episodes streamed through it, and it uses about 2 GB of GPU
memory.

| | |
|---|---|
| Python | 3.11.14 |
| torch | 2.7.0+cu126 (CUDA 12.6) |
| torchvision | 0.22.0+cu128 (from another CUDA index than torch; it only supplies the ResNet-34 definition here) |
| numpy, scipy | 1.26.0, 1.15.3 |
| opencv-python-headless | 4.11.0.86 |
| PyYAML, msgpack, websockets | 6.0.2, 1.1.2, 15.0.1 |
| pillow | 11.3.0 (a torchvision dependency) |
| cloudflared | 2026.10.3, for the public URL only |

numpy and msgpack differ from the challenge kit's own pins (2.4.6, 1.2.3), which the server does not use.

## Training, rehearsal and the camera loop

These also need the full environment in [`pip-freeze-evaluated-env.txt`](pip-freeze-evaluated-env.txt). The main parts:

| | |
|---|---|
| Isaac Sim | 5.1.0.0 (pip) |
| Isaac Lab | 0.50.5, editable, at commit `9d17b524536b341da09987ef765a0ad0501c64ca`. The working tree differs from it only in file modes, plus two untracked files that nothing imports |
| rsl-rl-lib | 3.1.2 |
| h5py, huggingface_hub, pytest | 3.15.1, 1.9.0, 9.0.2 |

## Machine

| | |
|---|---|
| OS | Ubuntu 22.10, Linux 5.19.0-46 |
| GPU | NVIDIA RTX A5000, 24 GB, driver 580.105.08 |
| CPU, memory | Intel i9-12900KS (24 threads), 62 GB |

## Challenge and data versions

| | |
|---|---|
| Track | `tower-sim-all`, version `20261013-v1-task45-10hz-reconnect`, 10 Hz |
| `track.json` used | sha256 `efbc4adbd1677bfb6751487c308d9658f35cc1070d77b1347752efc5732dedc8` |
| TOWER-Assets | revision `538d8b1fb582a4c3fc8c5a2fb2bf3a82931e9a4c` |
| TOWER-SimData | revision `1a59248e44866b1593e4ac8b851aa519c013220f` (same files as the docs-pinned `5ad6adc9`) |
