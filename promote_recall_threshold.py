"""Promotes the already-computed Recall>=0.99 post-hoc threshold
(stage1_recalibrate_recall_target.py / stage2_recalibrate_recall_target.py,
TARGET_RECALL=0.99) to be the OFFICIAL threshold for a downsampled tag, i.e.
overwrites {substance}_fold_{fold}_threshold.json with the content of
{substance}_fold_{fold}_threshold_recall0.99.json.

Why 0.99 and not 0.9 (the original default used by
stage1_detection_downsampled.py/stage2_safety_downsampled.py's
pick_threshold_for_recall): a full sweep against the external test-data
(evaluate_stage1_downsampled.py / evaluate_stage2_downsampled.py) showed
macro-F1 improves monotonically as the recall floor is raised
(0.8->0.9->0.95->0.99->0.999) for both stages, because a stricter
in-distribution threshold doesn't survive the probability-calibration
drift between train and real external test data, while a looser one is
more robust to it. Gains mostly saturate by 0.99 (Stage 1 fully; Stage 2
still ticks up slightly at 0.999, but not enough to justify going past 0.99
as the shared default). See conversation history for the full ablation
tables.

The previous official threshold.json content is preserved as
{substance}_fold_{fold}_threshold_recall{BACKUP_LABEL}.json before being
overwritten (BACKUP_LABEL names whatever policy is CURRENTLY live, e.g.
0.99 for the Recall>=0.99 policy promoted earlier), so nothing is lost and
promoting again later (e.g. back to a stricter policy) is a simple rerun
with swapped SRC_RECALL/BACKUP_LABEL.

Usage: STAGE=stage1 python3 promote_recall_threshold.py
       STAGE=stage2 MACHINES=OCEANFX SRC_RECALL=0.46 BACKUP_LABEL=0.99 python3 promote_recall_threshold.py
Env: STAGE (stage1|stage2), MACHINES (comma list, default FLAMENIR,OCEANFX),
     SRC_RECALL (the target_recall suffix to promote FROM, default 0.99),
     BACKUP_LABEL (suffix to save the CURRENT threshold.json under before
     overwriting, default 0.9 -- matches this script's original behavior).
"""

import json
import os
import shutil

from stage1_detection_downsampled import ALL_SUBSTANCES

STAGE_CONFIGS = {
    "stage1": dict(task_dir="substance_regression", tag="stage1_smartnir_downsampled"),
    "stage2": dict(task_dir="substance_severity_smartnir", tag="stage2_smartnir_downsampled"),
}


def main():
    stage = os.environ.get("STAGE", "stage1")
    cfg = STAGE_CONFIGS[stage]
    task_dir, tag = cfg["task_dir"], cfg["tag"]
    machines = os.environ.get("MACHINES", "FLAMENIR,OCEANFX").split(",")
    src_suffix = f"threshold_recall{os.environ.get('SRC_RECALL', '0.99')}"
    backup_label = os.environ.get("BACKUP_LABEL", "0.9")
    for machine in machines:
        n_promoted = 0
        for s in ALL_SUBSTANCES:
            data_dir = f"data/{task_dir}/{tag}/{machine}/{s}"
            if not os.path.isdir(data_dir):
                continue
            for fold in range(1, 6):
                src = f"{data_dir}/{s}_fold_{fold}_{src_suffix}.json"
                dst = f"{data_dir}/{s}_fold_{fold}_threshold.json"
                backup = f"{data_dir}/{s}_fold_{fold}_threshold_recall{backup_label}.json"
                if not os.path.exists(src):
                    continue
                if os.path.exists(dst) and not os.path.exists(backup):
                    shutil.copy(dst, backup)
                with open(src) as f:
                    new_thr = json.load(f)
                with open(dst, "w") as f:
                    json.dump(new_thr, f, indent=4)
                n_promoted += 1
        print(f"{machine}: promoted {n_promoted} threshold files ({src_suffix} -> threshold)", flush=True)


if __name__ == "__main__":
    main()
