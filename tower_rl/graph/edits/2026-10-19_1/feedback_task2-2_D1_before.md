# Rehearsal feedback: task2-2_cross_tower_transfer_2 / D1

2026-10-19T00:48 | episodes 8 | config `{"bundle": "results/bundles/grasp_left_v3_200", "device": "cpu", "episodes": 8, "finger_friction": [2.4, 2.0], "noise": {}, "perception_noise": false, "rate_hz": 10, "seed": 1000, "tier": "D1", "variant": "task2-2_cross_tower_transfer_2"}`

## Metrics

| metric | value | k / n | Wilson 95 % |
|---|---|---|---|
| TSR | 0.875 | 7 / 8 | 0.53-0.98 |
| GSR | 0.938 | 15 / 16 | 0.72-0.99 |
| PSR | 0.875 | 14 / 16 | 0.64-0.96 |
| LHPR | 0.906 | 29 / 32 | 0.76-0.97 |
| ODR | 0.030 | 1 / 33 | |

policy errors 0, mean duration 61.86 s; executor wall time per call mean 11.5 ms, max 149.6 ms; inference calls per episode mean 803

## By robustness case

| case | n | TSR | GSR | PSR | ODR |
|---|---|---|---|---|---|
| combined | 2 | 0.50 | 0.75 | 0.50 | 0.11 |
| high_friction | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| irregular_layers | 2 | 1.00 | 1.00 | 1.00 | 0.00 |
| lighting | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| shifted_assembly | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| slippery | 1 | 1.00 | 1.00 | 1.00 | 0.00 |

## Per-node outcomes

| node | attempts | ok | failed | timeout | duration p50/p90/max (s) | measurements p50/p90/max |
|---|---|---|---|---|---|---|
| t0_to_pregrasp | 10 | 9 | 0 | 1 | 5.8 / 6.8 / 15.1 | pos_err_mm 1.2 / 1.8 / 2.6; rot_err_deg 0.2 / 0.4 / 0.7; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 54.0 / 55.0 / 55.0 |
| t0_grasp | 9 | 8 | 0 | 1 | 2.4 / 5.0 / 15.0 | lift_mm 61.1 / 65.2 / 65.3; steps 25.0 / 26.0 / 26.0 |
| t0_to_preplace | 8 | 8 | 0 | 0 | 8.2 / 8.3 / 8.3 | pos_err_mm 1.5 / 1.8 / 1.8; rot_err_deg 0.3 / 0.3 / 0.3; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 77.5 / 78.0 / 78.0 |
| t0_place | 8 | 7 | 1 | 0 | 3.5 / 3.5 / 3.5 | place_xy_mm 0.8 / 2.9 / 5.9; yaw_deg 0.4 / 2.0 / 3.9 |
| park_left_1 | 7 | 7 | 0 | 0 | 2.8 / 2.8 / 2.8 |  |
| t1_to_pregrasp | 10 | 7 | 0 | 3 | 5.4 / 16.0 / 16.0 | pos_err_mm 2.6 / 2.6 / 2.6; rot_err_deg 0.4 / 0.4 / 0.4; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 50.0 / 50.0 / 50.0 |
| t1_grasp | 7 | 7 | 0 | 0 | 3.1 / 3.1 / 3.1 | lift_mm 64.6 / 64.8 / 64.8 |
| t1_to_preplace | 7 | 7 | 0 | 0 | 7.1 / 7.3 / 7.3 | pos_err_mm 3.1 / 3.5 / 4.2; rot_err_deg 0.5 / 0.6 / 0.7; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 66.0 / 68.0 / 68.0 |
| park_back_end | 1 | 1 | 0 | 0 | 5.5 / 5.5 / 5.5 |  |

## Failure signatures (most frequent first)

| node | status | why | count | example (seed, case, t) |
|---|---|---|---|---|
| t1_to_pregrasp | timeout |  | 3 | (1100001, 'combined', 67.9) |
| t0_place | failed |  | 1 | (1100001, 'combined', 20.1) |
| t0_grasp | timeout |  | 1 | (1100001, 'combined', 36.8) |
| t0_to_pregrasp | timeout |  | 1 | (1100001, 'combined', 51.9) |

## Disturbances by operation window (index: G0, P0, G1, P1, ..., final)

1: 1

## Failed episodes (seeds for videos)

1100001(combined)
