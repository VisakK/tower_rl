# Rehearsal feedback: task1-3_tower_transfer_3 / D1

2026-10-19T00:14 | episodes 8 | config `{"bundle": null, "device": "cpu", "episodes": 8, "finger_friction": [2.4, 2.0], "perception_noise": false, "rate_hz": 10, "seed": 1000, "tier": "D1", "variant": "task1-3_tower_transfer_3"}`

## Metrics

| metric | value | k / n | Wilson 95 % |
|---|---|---|---|
| TSR | 0.875 | 7 / 8 | 0.53-0.98 |
| GSR | 0.958 | 46 / 48 | 0.86-0.99 |
| PSR | 0.875 | 42 / 48 | 0.75-0.94 |
| LHPR | 0.917 | 88 / 96 | 0.84-0.96 |
| ODR | 0.093 | 9 / 97 | |

policy errors 0, mean duration 134.04 s; executor wall time per call mean 10.8 ms, max 138.8 ms; inference calls per episode mean 1117

## By robustness case

| case | n | TSR | GSR | PSR | ODR |
|---|---|---|---|---|---|
| combined | 2 | 0.50 | 0.83 | 0.50 | 0.36 |
| high_friction | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| irregular_layers | 2 | 1.00 | 1.00 | 1.00 | 0.00 |
| lighting | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| shifted_assembly | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| slippery | 1 | 1.00 | 1.00 | 1.00 | 0.00 |

## Per-node outcomes

| node | attempts | ok | failed | timeout | duration p50/p90/max (s) | measurements p50/p90/max |
|---|---|---|---|---|---|---|
| t0_to_pregrasp | 10 | 8 | 0 | 2 | 5.7 / 16.0 / 16.0 | pos_err_mm 1.1 / 1.1 / 1.1; rot_err_deg 0.2 / 0.2 / 0.2; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 53.0 / 53.0 / 53.0 |
| t0_grasp | 8 | 8 | 0 | 0 | 3.1 / 3.1 / 3.2 | lift_mm 65.7 / 66.5 / 67.5 |
| t0_to_preplace | 8 | 8 | 0 | 0 | 8.2 / 8.3 / 8.4 | pos_err_mm 3.2 / 3.4 / 3.5; rot_err_deg 1.1 / 1.3 / 1.3; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 77.5 / 78.3 / 79.0 |
| t0_place | 8 | 7 | 1 | 0 | 3.5 / 3.5 / 3.5 | place_xy_mm 0.9 / 597.7 / 1989.9; yaw_deg 0.7 / 11.3 / 34.9 |
| t1_to_pregrasp | 11 | 9 | 0 | 2 | 5.1 / 16.0 / 16.0 | pos_err_mm 1.3 / 2.6 / 6.0; rot_err_deg 0.2 / 0.6 / 1.7; max_block_disp_mm 0.0 / 208453.5 / 1042267.7; rows 47.0 / 47.0 / 47.0 |
| t1_grasp | 9 | 9 | 0 | 0 | 3.1 / 3.1 / 3.1 | lift_mm 65.2 / 66.8 / 70.0 |
| t1_to_preplace | 9 | 9 | 0 | 0 | 8.2 / 8.3 / 8.4 | pos_err_mm 1.2 / 1.3 / 1.3; rot_err_deg 0.2 / 0.2 / 0.2; max_block_disp_mm 0.0 / 1253139.6 / 6265698.2; rows 77.0 / 78.2 / 79.0 |
| t1_place | 9 | 7 | 2 | 0 | 3.5 / 3.5 / 3.5 | place_xy_mm 3.1 / 9.2 / 11.9; yaw_deg 0.4 / 0.5 / 0.6 |
| park_left_2 | 7 | 7 | 0 | 0 | 2.6 / 2.6 / 2.6 |  |
| t2_to_pregrasp | 10 | 9 | 0 | 1 | 5.2 / 6.2 / 15.1 | pos_err_mm 2.6 / 6.0 / 19.1; rot_err_deg 0.4 / 0.9 / 2.9; max_block_disp_mm 0.0 / 3640730.8 / 4540652.8; rows 48.0 / 48.0 / 48.0 |
| t2_grasp | 9 | 8 | 0 | 1 | 3.1 / 5.5 / 15.1 | lift_mm 63.3 / 63.7 / 63.8 |
| t2_to_preplace | 8 | 7 | 0 | 1 | 7.4 / 11.4 / 20.2 | pos_err_mm 3.2 / 3.7 / 4.2; rot_err_deg 0.5 / 0.6 / 0.7; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 69.0 / 70.4 / 71.0 |
| t2_place | 7 | 7 | 0 | 0 | 3.5 / 3.5 / 3.5 | place_xy_mm 1.4 / 1.8 / 2.3; yaw_deg 0.3 / 0.4 / 0.4 |
| t3_to_pregrasp | 9 | 9 | 0 | 0 | 4.7 / 4.8 / 4.8 | pos_err_mm 2.5 / 5.0 / 11.6; rot_err_deg 0.4 / 0.8 / 2.0; max_block_disp_mm 0.0 / 4997564.1 / 6195820.3; rows 43.0 / 44.0 / 44.0 |
| t3_grasp | 9 | 8 | 0 | 1 | 3.1 / 5.5 / 15.1 | lift_mm 63.2 / 63.3 / 63.4 |
| t3_to_preplace | 8 | 7 | 0 | 1 | 7.4 / 11.4 / 20.2 | pos_err_mm 3.2 / 3.6 / 4.1; rot_err_deg 0.5 / 0.6 / 0.7; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 69.0 / 70.4 / 71.0 |
| t3_place | 7 | 7 | 0 | 0 | 3.5 / 3.5 / 3.5 | place_xy_mm 1.6 / 1.8 / 2.1; yaw_deg 0.1 / 0.2 / 0.2 |
| park_back_4 | 7 | 7 | 0 | 0 | 2.4 / 2.4 / 2.4 |  |
| t4_to_pregrasp | 7 | 7 | 0 | 0 | 6.1 / 6.1 / 6.1 | pos_err_mm 1.4 / 1.4 / 1.5; rot_err_deg 0.2 / 0.2 / 0.2; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 57.0 / 57.0 / 57.0 |
| t4_grasp | 7 | 7 | 0 | 0 | 3.1 / 3.1 / 3.1 | lift_mm 66.5 / 66.9 / 67.0 |
| t4_to_preplace | 7 | 7 | 0 | 0 | 7.5 / 7.6 / 7.7 | pos_err_mm 3.5 / 3.6 / 3.6; rot_err_deg 2.0 / 2.1 / 2.1; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 70.0 / 71.4 / 72.0 |
| t4_place | 7 | 7 | 0 | 0 | 3.5 / 3.5 / 3.5 | place_xy_mm 2.6 / 3.2 / 3.6; yaw_deg 1.4 / 1.8 / 2.0 |
| t5_to_pregrasp | 7 | 7 | 0 | 0 | 5.1 / 5.1 / 5.1 | pos_err_mm 1.1 / 1.4 / 1.7; rot_err_deg 0.2 / 0.2 / 0.3; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 47.0 / 47.0 / 47.0 |
| t5_grasp | 7 | 7 | 0 | 0 | 3.1 / 3.1 / 3.1 | lift_mm 66.5 / 66.7 / 66.8 |
| t5_to_preplace | 7 | 7 | 0 | 0 | 7.5 / 7.6 / 7.6 | pos_err_mm 0.5 / 0.5 / 0.6; rot_err_deg 0.1 / 0.1 / 0.1; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 70.0 / 71.0 / 71.0 |

## Failure signatures (most frequent first)

| node | status | why | count | example (seed, case, t) |
|---|---|---|---|---|
| t1_place | failed |  | 2 | (1000001, 'combined', 40.7) |
| t0_to_pregrasp | timeout |  | 2 | (1100001, 'combined', 36.2) |
| t1_to_pregrasp | timeout |  | 2 | (1100001, 'combined', 68.2) |
| t0_place | failed |  | 1 | (1100001, 'combined', 20.2) |
| t2_to_preplace | timeout |  | 1 | (1100001, 'combined', 129.0) |
| t2_grasp | timeout |  | 1 | (1100001, 'combined', 147.1) |
| t2_to_pregrasp | timeout |  | 1 | (1100001, 'combined', 162.2) |
| t3_to_preplace | timeout |  | 1 | (1100001, 'combined', 189.7) |
| t3_grasp | timeout |  | 1 | (1100001, 'combined', 207.5) |

## Disturbances by operation window (index: G0, P0, G1, P1, ..., final)

2: 1, 3: 1, 4: 1, 5: 2, 6: 1, 7: 3

## Failed episodes (seeds for videos)

1100001(combined)
