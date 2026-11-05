# `student_v1_2` as evaluated

Tag `eval/student_v1_2` holds the exact code that served `student_v1_2` in the TOWER challenge evaluation.
[`REVIEW.md` on `main`](https://github.com/VisakK/tower_rl/blob/main/REVIEW.md) covers all submissions and how this code differs from `main`.

| | |
|---|---|
| Policy ID | `tower-student-v1-2` |
| Artifact identity | `sha256:f726c71a453da23943757cf9fb8506d101f83b196b22770f966073a5c3279aa9`, the sha256 of `student_v1_2.tar.gz`, which the server advertises |
| Plugin version | `2` |
| Track | `tower-sim-all`, version `20261013-v1-task45-10hz-reconnect`, 10 Hz |
| Code snapshot | `tower_rl_snapshots/server_v1_2` on the development machine |
| Bundle created | 2026-10-28T10:54:17 PDT (its `MANIFEST.json`) |

## How this commit relates to what ran

**No git commit existed when this ran.** The server ran from a frozen copy of the code outside git
(`scripts/make_snapshot.sh`), and the bundle's MANIFEST records `git_commit: f5e8cce` with `git_dirty: true`. This commit
was made on 2026-11-05 from that frozen copy, so its date is later than the evaluation.

- `tower_rl/`, `scripts/`, `configs/` and `pytest.ini` (239 files) are byte-identical to the snapshot, without
  `__pycache__`.
- Three files are added: `assets_m/tower_station_m.json`, `.gitignore` and this README.
- The snapshot's `data`, `assets_m` and `results` were symlinks into the development repository, and the bundle and
  `track.json` were passed as arguments. `assets_m/tower_station_m.json` is the one other file the server read through
  those links. It is the same file (sha256 `33dd726c340eaa7b3877a486b0380ad8e9bebd2cf0851d18e4b28823802b991d`, last modified 2026-10-18 22:05 PDT). The server reads
  only its `cameras` section, so the local paths elsewhere in the file are not used.

Evidence for the freeze date, from the development machine: the newest file in the snapshot is from 2026-10-28 10:10 PDT, and the
oldest Python cache in its `tower_rl/server/` is from 2026-10-28 10:51 PDT. The server logs were shared with the reviewers.

## What ran

Two evaluation attempts were cut off by hard resets of the development machine, so neither finished. Every process
served this same code and bundle:

- Attempt 1: served 2026-10-28 11:09:03 PDT, evaluation started 19:55:24, machine reset 2026-10-29 00:10:55.
- Attempt 2: served 2026-10-29 05:28:21 PDT, evaluation started 16:06:23, machine reset 16:41:15.
- Served again 2026-10-29 16:54:23 PDT as a fallback. No evaluation ran on this process.

The command, from the snapshot directory (`run.json`):

```bash
python scripts/serve_student.py --track data/challenge/track.json --bundle /home/visakii/Documents/tower_rl/results/bundles/student_v1_2 --name official_v1_2 --port 9003 --policy_id tower-student-v1-2 --plugin_version 2
```

The official wrappers (tmux, Cloudflare quick tunnel) are in `reproduce/evaluated/official/` on `main`.

## Serve it

You need Python 3.11, a CUDA GPU with about 2 GB free for the pose network, and these packages
(`reproduce/environment/requirements-inference.txt` on `main` installs them). Isaac Sim and Isaac Lab are not needed to
serve.

```
torch==2.7.0+cu126  torchvision==0.22.0+cu128  numpy==1.26.0  scipy==1.15.3
opencv-python-headless==4.11.0.86  PyYAML==6.0.2  msgpack==1.1.2  websockets==15.0.1
```

The bundle tar was shared with the reviewers separately, and `track.json` comes from the challenge site.

```bash
git clone https://github.com/VisakK/tower_rl.git && cd tower_rl
git checkout eval/student_v1_2
sha256sum student_v1_2.tar.gz        # must print f726c71a453da23943757cf9fb8506d101f83b196b22770f966073a5c3279aa9
mkdir -p results/bundles && tar xzf student_v1_2.tar.gz -C results/bundles
python scripts/serve_student.py --track <track.json> --bundle results/bundles/student_v1_2 --name review \
    --port 9003 --policy_id tower-student-v1-2 --plugin_version 2 --artifact_identity sha256:f726c71a453da23943757cf9fb8506d101f83b196b22770f966073a5c3279aa9

# second terminal: the qualification-like checks (every variant, two resets, zero-valued images)
python scripts/probe_server.py ws://127.0.0.1:9003 --track <track.json>
```

The evaluator needs a public `wss://` URL, for example `cloudflared tunnel --no-autoupdate --url
http://127.0.0.1:9003`. Logs go to `results/student/server/review/`.
