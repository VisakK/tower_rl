#!/usr/bin/env bash
# Stage 3 (reproduce/README.md §5): the pose estimator kp_v2 and the orientation references, as built for student_v1_3
# (docs/Pose-estimator-v1.MD). Needs stage 2's teacher_t45_final (the renders run the teacher) and a GPU (rendering,
# labelling, training). Recorded times on an RTX A5000: renders ~3 h, organizer frames ~1 h (network), labels ~30 min,
# kp_v1 ~100 min, kp_v2 ~45 min, evaluation ~25 min.
# The order is the original one and matters: kp_v1 trains on the organizer frames' raw foundation labels; the
# foundation corrections come after it; the second batch of training episodes is corrected before it is labelled.
# Not bit-exact: rendering, and training's augmentation (seeded from os.urandom); a retrained kp_v2 is equivalent, not
# identical. Outputs: results/perception/models/kp_v2/model_last.pt, results/perception/orient_refs.npz.
# Usage: reproduce/3_perception.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
VARIANTS="task1-2_tower_transfer_2 task1-3_tower_transfer_3 task2-2_cross_tower_transfer_2 task2-3_cross_tower_transfer_3
task3-2_vertical_stacking_2 task3-3_vertical_stacking_3 task4-2_full_tower_pushdown_2 task4-3_full_tower_pushdown_3
task5-2_tower_insertion_2 task5-3_tower_insertion_3"
TEACHER=${TEACHER:-results/bundles/teacher_t45_final}
# 1. renders: the teacher runs episodes, the four cameras are captured at random rows (28k train, 7k validation frames)
for v in $VARIANTS; do
  run "$PY" scripts/render_perception_data.py --headless --device cpu --variant $v --tiers D0 D1 --episodes 16 \
      --num_envs 4 --seed 20000 --capture_p 0.1 --bundle "$TEACHER" --out data/perception/r1
done
for v in $VARIANTS; do
  run "$PY" scripts/render_perception_data.py --headless --device cpu --variant $v --tiers D0 D1 --episodes 4 \
      --num_envs 4 --seed 90000 --capture_p 0.1 --bundle "$TEACHER" --out data/perception/r1_val
done
# 2. organizer frames (the evaluator's renderer) of the Step 1 subset: validation = subset episodes 0-2 (24 frames
#    each), training batch 1 = episodes 3-7 (40 frames each); then occlusion-aware labels from Isaac depth
for v in $VARIANTS; do
  run "$PY" scripts/fetch_dataset_frames.py --variant $v --episodes 3 --frames 24 --out data/perception/dataset_val
  run "$PY" scripts/fetch_dataset_frames.py --variant $v --skip 3 --episodes 5 --frames 40 --out data/perception/dataset_train
done
for root in data/perception/dataset_train data/perception/dataset_val; do
  for v in $VARIANTS; do run "$PY" scripts/label_dataset_frames.py --headless --variant $v --root $root; done
done
# 3. kp_v1: ImageNet ResNet-34 U-Net, renders + organizer frames (weight 3)
run "$PY" scripts/train_perception.py --train data/perception/r1 data/perception/dataset_train:3 \
    --val data/perception/r1_val --iters 25000 --batch 12 --crop 512 384 --lr 1e-3 --workers 10 --val_every 2500 \
    --val_images 400 --out results/perception/models/kp_v1
# 4. the organizer episodes' foundation pads sit a few mm off their recorded poses: registered and corrected
run "$PY" scripts/correct_dataset_foundations.py --root data/perception/dataset_train
run "$PY" scripts/correct_dataset_foundations.py --root data/perception/dataset_val
# 5. training batch 2 = subset episodes 8-9, corrected (only the new shards) and then everything relabelled
for v in $VARIANTS; do
  run "$PY" scripts/fetch_dataset_frames.py --variant $v --skip 8 --episodes 2 --frames 40 --out data/perception/dataset_train
done
run "$PY" scripts/correct_dataset_foundations.py --root data/perception/dataset_train
for root in data/perception/dataset_train data/perception/dataset_val; do
  for v in $VARIANTS; do run "$PY" scripts/label_dataset_frames.py --headless --variant $v --root $root; done
done
# 6. kp_v2: kp_v1's weights fine-tuned on the corrected data (new optimizer, cosine schedule); model_last is deployed
run "$PY" scripts/train_perception.py --train data/perception/r1 data/perception/dataset_train:3 \
    --val data/perception/r1_val --resume results/perception/models/kp_v1/model_last.pt --iters 12000 --batch 12 \
    --crop 512 384 --lr 5e-4 --workers 10 --val_every 3000 --val_images 400 --out results/perception/models/kp_v2
# 7. accuracy (student_v1_3's kp_v2, organizer validation frames: tracking error median 0.198 mm, first call 0.183 mm)
CK=results/perception/models/kp_v2/model_last.pt
E=results/perception/eval/kp_v2
mkdir -p $E
run "$PY" scripts/eval_perception.py --data data/perception/dataset_val --checkpoint $CK --mode prior \
    --noise 0.0015 1.0 0.5 --frames 400 --out $E/dataset_track.json
run "$PY" scripts/eval_perception.py --data data/perception/dataset_val --checkpoint $CK --mode init --max_t 2.0 \
    --frames 100 --out $E/dataset_init.json
run "$PY" scripts/eval_perception_sequence.py --hdf5 data/episodes_full/data/*/*.hdf5 --checkpoint $CK \
    --out $E/sequences.json --foundation_corrections data/perception/dataset_val/foundation_corrections.json
# 8. orientation references for the fixed cameras (first frame of each full episode; byte-identical rebuild)
run "$PY" scripts/build_orient_refs.py --out results/perception/orient_refs.npz
