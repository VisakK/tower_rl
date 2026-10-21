# Rehearsal feedback: task1-3_tower_transfer_3 / D1

2026-10-19T05:26 | episodes 48 | config `{"bundle": "results/bundles/probe_v4grasps", "device": "cpu", "episodes": 48, "finger_friction": [2.4, 2.0], "noise": {}, "perception_noise": false, "rate_hz": 10, "seed": 9000, "tier": "D1", "variant": "task1-3_tower_transfer_3"}`

## Metrics

| metric | value | k / n | Wilson 95 % |
|---|---|---|---|
| TSR | 0.833 | 40 / 48 | 0.70-0.91 |
| GSR | 0.983 | 283 / 288 | 0.96-0.99 |
| PSR | 0.931 | 268 / 288 | 0.90-0.95 |
| LHPR | 0.957 | 551 / 576 | 0.94-0.97 |
| ODR | 0.067 | 39 / 584 | |

policy errors 0, mean duration 128.6 s; executor wall time per call mean 15.8 ms, max 655.3 ms; inference calls per episode mean 949
closest arm-arm approach per episode (m): min 0.046, p10 0.165, median 0.171

## By robustness case

| case | n | TSR | GSR | PSR | ODR |
|---|---|---|---|---|---|
| combined | 6 | 0.67 | 1.00 | 0.83 | 0.12 |
| heavy | 8 | 0.50 | 0.90 | 0.77 | 0.21 |
| high_friction | 4 | 1.00 | 1.00 | 1.00 | 0.00 |
| irregular_layers | 4 | 1.00 | 1.00 | 1.00 | 0.00 |
| lighting | 5 | 1.00 | 1.00 | 1.00 | 0.00 |
| lightweight | 8 | 1.00 | 1.00 | 1.00 | 0.00 |
| nominal | 7 | 1.00 | 1.00 | 1.00 | 0.00 |
| shifted_assembly | 2 | 1.00 | 1.00 | 1.00 | 0.00 |
| slippery | 1 | 1.00 | 1.00 | 1.00 | 0.00 |
| uneven_weights | 3 | 0.33 | 1.00 | 0.83 | 0.24 |

## Per-node outcomes

| node | attempts | ok | failed | timeout | duration p50/p90/max (s) | measurements p50/p90/max |
|---|---|---|---|---|---|---|
| t0_to_pregrasp | 48 | 48 | 0 | 0 | 5.7 / 5.7 / 5.7 | pos_err_mm 0.9 / 1.0 / 1.3; rot_err_deg 0.2 / 0.2 / 0.2; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 53.0 / 53.0 / 53.0 |
| t0_grasp | 48 | 48 | 0 | 0 | 2.4 / 2.4 / 2.4 | lift_mm 62.9 / 64.9 / 65.1; steps 24.5 / 25.0 / 25.0; grasp_rot_deg 1.1 / 1.7 / 2.1 |
| t0_to_preplace | 48 | 48 | 0 | 0 | 8.7 / 8.7 / 8.8 | pos_err_mm 3.0 / 3.1 / 3.3; rot_err_deg 1.1 / 1.1 / 1.2; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 81.5 / 82.0 / 83.0 |
| t0_place | 48 | 47 | 1 | 0 | 1.8 / 1.8 / 1.8 | place_xy_mm 2.4 / 4.0 / 5.1; yaw_deg 0.6 / 1.9 / 2.7; steps 18.0 / 18.0 / 18.0 |
| t1_to_pregrasp | 54 | 54 | 0 | 0 | 5.3 / 5.3 / 5.3 | pos_err_mm 1.5 / 1.6 / 7.2; rot_err_deg 0.3 / 0.3 / 1.8; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 49.0 / 49.0 / 49.0; goal_adjust_mm 16.2 / 26.5 / 29.1; goal_adjust_deg 4.4 / 7.2 / 7.9 |
| t1_grasp | 54 | 53 | 0 | 1 | 2.3 / 2.5 / 15.0 | lift_mm 61.7 / 64.9 / 65.1; steps 24.0 / 25.0 / 28.0; grasp_rot_deg 0.6 / 5.7 / 19.9 |
| t1_to_preplace | 53 | 51 | 2 | 0 | 8.4 / 8.5 / 8.6 | pos_err_mm 1.2 / 1.3 / 1.4; rot_err_deg 0.2 / 0.2 / 0.3; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 79.0 / 80.0 / 81.0 |
| t1_place | 51 | 47 | 4 | 0 | 1.9 / 1.9 / 2.5 | place_xy_mm 2.8 / 4.0 / 6.7; yaw_deg 1.3 / 1.9 / 16.4; steps 19.0 / 19.0 / 25.0 |
| park_left_2 | 48 | 48 | 0 | 0 | 2.4 / 2.5 / 6.9 |  |
| t2_to_pregrasp | 50 | 49 | 1 | 0 | 5.2 / 5.2 / 5.8 | pos_err_mm 2.6 / 2.7 / 3.6; rot_err_deg 0.4 / 0.5 / 0.6; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 48.0 / 48.0 / 54.0; goal_adjust_mm 80436.5 / 80436.5 / 80436.5; goal_adjust_deg 69.1 / 69.1 / 69.1 |
| t2_grasp | 49 | 49 | 0 | 0 | 2.0 / 2.0 / 2.4 | lift_mm 66.9 / 70.2 / 94.0; steps 21.0 / 21.0 / 25.0; grasp_rot_deg 1.3 / 1.9 / 12.8 |
| t2_to_preplace | 49 | 49 | 0 | 0 | 7.1 / 7.2 / 7.3 | pos_err_mm 3.3 / 4.1 / 4.7; rot_err_deg 0.5 / 0.6 / 0.7; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 66.0 / 67.0 / 68.0 |
| t2_place | 49 | 47 | 2 | 0 | 3.6 / 3.6 / 3.6 | place_xy_mm 1.6 / 2.5 / 7.6; yaw_deg 0.2 / 0.5 / 1.4 |
| t3_to_pregrasp | 50 | 47 | 3 | 0 | 4.7 / 4.8 / 4.9 | pos_err_mm 2.4 / 2.5 / 2.5; rot_err_deg 0.4 / 0.4 / 0.4; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 43.0 / 44.0 / 45.0; goal_adjust_mm 40.5 / 40.5 / 40.5; goal_adjust_deg 1.5 / 1.5 / 1.5 |
| t3_grasp | 47 | 47 | 0 | 0 | 2.0 / 2.0 / 2.0 | lift_mm 64.4 / 69.1 / 69.7; steps 21.0 / 21.0 / 21.0; grasp_rot_deg 0.6 / 1.0 / 1.7 |
| t3_to_preplace | 47 | 47 | 0 | 0 | 7.1 / 7.1 / 7.2 | pos_err_mm 3.1 / 4.0 / 4.7; rot_err_deg 0.5 / 0.6 / 0.7; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 66.0 / 66.0 / 67.0 |
| t3_place | 47 | 47 | 0 | 0 | 3.6 / 3.6 / 3.6 | place_xy_mm 2.1 / 2.6 / 4.1; yaw_deg 0.1 / 0.4 / 0.7 |
| park_back_4 | 48 | 48 | 0 | 0 | 2.4 / 2.4 / 2.4 |  |
| t4_to_pregrasp | 50 | 47 | 3 | 0 | 6.0 / 6.1 / 6.1 | pos_err_mm 1.4 / 1.4 / 1.4; rot_err_deg 0.2 / 0.2 / 0.2; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 56.0 / 57.0 / 57.0; goal_adjust_mm 51.4 / 76276.3 / 95332.5; goal_adjust_deg 13.9 / 34.3 / 39.4 |
| t4_grasp | 47 | 47 | 0 | 0 | 2.3 / 2.4 / 2.4 | lift_mm 60.3 / 64.4 / 64.8; steps 24.0 / 25.0 / 25.0; grasp_rot_deg 0.3 / 0.8 / 1.6 |
| t4_to_preplace | 47 | 46 | 1 | 0 | 8.2 / 8.2 / 8.3 | pos_err_mm 3.5 / 3.5 / 3.7; rot_err_deg 1.8 / 1.9 / 1.9; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 77.0 / 77.5 / 78.0; goal_adjust_mm 0.8 / 0.8 / 0.9; goal_adjust_deg 0.5 / 0.5 / 0.5 |
| t4_place | 46 | 41 | 5 | 0 | 1.9 / 1.9 / 1.9 | place_xy_mm 1.4 / 4.4 / 7.8; yaw_deg 0.4 / 1.4 / 3.1; steps 19.0 / 19.0 / 19.0 |
| t5_to_pregrasp | 57 | 54 | 3 | 0 | 5.1 / 5.2 / 7.7 | pos_err_mm 1.6 / 1.7 / 1.9; rot_err_deg 0.3 / 0.3 / 0.3; max_block_disp_mm 0.0 / 0.0 / 488984.8; rows 47.0 / 48.0 / 73.0; goal_adjust_mm 118.7 / 67975.4 / 97056.8; goal_adjust_deg 5.9 / 122.7 / 172.7 |
| t5_grasp | 54 | 49 | 0 | 5 | 2.3 / 2.7 / 15.0 | lift_mm 60.5 / 64.9 / 64.9; steps 24.0 / 25.0 / 29.0; grasp_rot_deg 0.5 / 1.6 / 7.2 |
| t5_to_preplace | 48 | 45 | 3 | 0 | 7.9 / 8.0 / 8.1 | pos_err_mm 0.4 / 0.4 / 1.4; rot_err_deg 0.1 / 0.1 / 0.3; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 74.0 / 75.0 / 76.0 |
| t5_place | 43 | 39 | 4 | 0 | 1.8 / 1.9 / 1.9 | place_xy_mm 1.8 / 2.5 / 29.2; yaw_deg 0.6 / 2.0 / 11.7; steps 18.0 / 18.8 / 19.0 |
| park_left_end | 7 | 7 | 0 | 0 | 4.3 / 6.9 / 6.9 |  |
| t4_alt_park_left | 5 | 5 | 0 | 0 | 2.7 / 2.8 / 2.8 |  |
| t4_to_pregrasp_alt | 5 | 5 | 0 | 0 | 6.1 / 6.3 / 6.4 | pos_err_mm 3.7 / 3.9 / 3.9; rot_err_deg 0.7 / 0.7 / 0.8; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 57.0 / 58.8 / 60.0; goal_adjust_mm 0.3 / 0.4 / 0.4; goal_adjust_deg 0.6 / 0.6 / 0.6 |
| t4_grasp_alt | 5 | 1 | 0 | 4 | 15.0 / 15.0 / 15.0 | lift_mm 65.3 / 65.3 / 65.3; steps 97.0 / 97.0 / 97.0; grasp_rot_deg 6.3 / 6.3 / 6.3 |
| t4_alt_park_back | 5 | 5 | 0 | 0 | 6.4 / 7.0 / 7.1 |  |
| t0_alt_park_left | 1 | 1 | 0 | 0 | 2.9 / 2.9 / 2.9 |  |
| t0_to_pregrasp_alt | 1 | 1 | 0 | 0 | 6.0 / 6.0 / 6.0 | pos_err_mm 3.3 / 3.3 / 3.3; rot_err_deg 0.7 / 0.7 / 0.7; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 56.0 / 56.0 / 56.0; goal_adjust_mm 3.8 / 3.8 / 3.8; goal_adjust_deg 1.4 / 1.4 / 1.4 |
| t0_grasp_alt | 1 | 0 | 0 | 1 | 15.0 / 15.0 / 15.0 |  |
| t0_alt_park_back | 1 | 1 | 0 | 0 | 6.9 / 6.9 / 6.9 |  |
| t4_to_preplace_alt | 1 | 1 | 0 | 0 | 5.3 / 5.3 / 5.3 | pos_err_mm 4.6 / 4.6 / 4.6; rot_err_deg 0.7 / 0.7 / 0.7; max_block_disp_mm 0.0 / 0.0 / 0.0; rows 48.0 / 48.0 / 48.0 |
| t4_place_alt | 1 | 1 | 0 | 0 | 3.6 / 3.6 / 3.6 | place_xy_mm 1.1 / 1.1 / 1.1; yaw_deg 1.8 / 1.8 / 1.8 |

## Failure signatures (most frequent first)

| node | status | why | count | example (seed, case, t) |
|---|---|---|---|---|
| t5_grasp | timeout |  | 5 | (9000003, 'uneven_weights', 133.5) |
| t4_place | failed |  | 5 | (9200000, 'combined', 94.9) |
| t4_grasp_alt | timeout |  | 4 | (9200000, 'combined', 118.7) |
| t5_place | failed |  | 4 | (9200003, 'uneven_weights', 140.9) |
| t1_place | failed |  | 4 | (9300000, 'heavy', 43.1) |
| t5_to_preplace | failed | dropped | 3 | (9000003, 'uneven_weights', 105.5) |
| t5_to_pregrasp | failed | pregrasp unreachable | 3 | (9300000, 'heavy', 164.1) |
| t4_to_pregrasp | failed | pregrasp unreachable | 3 | (9300002, 'heavy', 101.6) |
| t3_to_pregrasp | failed | pregrasp unreachable | 3 | (9400001, 'heavy', 60.1) |
| t1_to_preplace | failed | dropped | 2 | (9300000, 'heavy', 33.7) |
| t2_place | failed |  | 2 | (9700001, 'combined', 108.3) |
| t4_to_preplace | failed | dropped | 1 | (9300002, 'heavy', 101.6) |

## Disturbances by operation window (index: G0, P0, G1, P1, ..., final)

1: 4, 2: 1, 3: 5, 9: 14, 11: 15

## Failed episodes (seeds for videos)

9000003(uneven_weights), 9200000(combined), 9200003(uneven_weights), 9300000(heavy), 9300002(heavy), 9300003(heavy), 9400001(heavy), 9700001(combined)
