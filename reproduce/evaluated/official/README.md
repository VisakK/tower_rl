# The scripts that served the official evaluations

Verbatim copies of files from the development machine. They kept to its paths (`/home/visakii/...`) and tools (tmux,
`systemd-inhibit`, a Cloudflare quick tunnel), so they are a record, not something to run elsewhere. To serve a submission
yourself, use [`../serve_evaluated.sh`](../serve_evaluated.sh) or the README of its `eval/*` tag.

| File | What it is |
|---|---|
| `serve_official.sh` | Starts the server from a code snapshot in a tmux session, then a quick tunnel, and prints the model URL. Every official endpoint ran through it |
| `stop_official.sh` | Stops an endpoint |
| `build_v1_2.sh`, `build_v1_3.sh` | Freeze the bundle, check it against the previous version, make the code snapshot, serve it, and run the probes locally and through the tunnel. `student_v1` was frozen by `scripts/freeze_student_bundle.py` and served with `serve_official.sh`'s defaults |
| `run_student_v1.json` | The server's `run.json` for `student_v1` |
| `run_student_v1_2_attempt1.json`, `run_student_v1_2_attempt2.json` | `student_v1_2`, one per evaluation attempt, both cut off by machine resets |
| `run_student_v1_2_fallback.json` | `student_v1_2` served again as a fallback; no evaluation ran on it |
| `run_student_v1_3_attempt1.json` | `student_v1_3`'s attempt cut off by a machine reset |
| `run_student_v1_3.json` | `student_v1_3`'s completed evaluation |

Each `run.json` records the exact arguments, the artifact identity, the policy ID, the plugin version and the track version.
`serve_official.sh` also holds the code path (`SNAP`) that `run.json` does not record: the server's working directory was
the submission's snapshot, now its `eval/*` tag.
