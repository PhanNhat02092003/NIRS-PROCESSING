#!/bin/bash
# Waits for the already-running Mango GuidedDCNet quality job to finish,
# then runs the remaining quality-grade benchmark trainings back-to-back
# (Grainit SMART-NIR/GuidedDCNet, Rapeseed SMART-NIR/GuidedDCNet), appending
# each one's result to its dataset's benchmark/quality/*.csv as soon as it
# finishes -- so nothing needs to be launched by hand between runs.
set -e
ROOT=/home/phannhat/NIRS-PROCESSING/additional_experiments

echo "[queue] waiting for Mango GuidedDCNet quality (PID 1875502) to finish..."
while kill -0 1875502 2>/dev/null; do sleep 5; done
echo "[queue] Mango GuidedDCNet quality finished."
python3 "$ROOT/append_quality_result.py" \
  --history-glob "$ROOT/Mango/history/category_classification_quality_guideddcnet/Mango/guideddcnet_quality_fold*.json" \
  --csv "$ROOT/Mango/benchmark/quality/Mango.csv" --method "GuidedDCNet"

echo "[queue] === Starting Grainit SMART-NIR quality ==="
cd "$ROOT/Grainit" && python3 grainit_quality_smartnir_engine.py
python3 "$ROOT/append_quality_result.py" \
  --history-glob "$ROOT/Grainit/history/category_classification_quality_smartnir/Grainit/smart_nir_quality_fold*.json" \
  --csv "$ROOT/Grainit/benchmark/quality/Grainit.csv" --method "SMART-NIR"

echo "[queue] === Starting Grainit GuidedDCNet quality ==="
cd "$ROOT/Grainit" && python3 grainit_quality_guideddcnet_engine.py
python3 "$ROOT/append_quality_result.py" \
  --history-glob "$ROOT/Grainit/history/category_classification_quality_guideddcnet/Grainit/guideddcnet_quality_fold*.json" \
  --csv "$ROOT/Grainit/benchmark/quality/Grainit.csv" --method "GuidedDCNet"

echo "[queue] === Starting Rapeseed SMART-NIR quality ==="
cd "$ROOT/Rapeseed" && python3 rapeseed_quality_smartnir_engine.py
python3 "$ROOT/append_quality_result.py" \
  --history-glob "$ROOT/Rapeseed/history/category_classification_quality_smartnir/Rapeseed/smart_nir_quality_fold*.json" \
  --csv "$ROOT/Rapeseed/benchmark/quality/Rapeseed.csv" --method "SMART-NIR"

echo "[queue] === Starting Rapeseed GuidedDCNet quality ==="
cd "$ROOT/Rapeseed" && python3 rapeseed_quality_guideddcnet_engine.py
python3 "$ROOT/append_quality_result.py" \
  --history-glob "$ROOT/Rapeseed/history/category_classification_quality_guideddcnet/Rapeseed/guideddcnet_quality_fold*.json" \
  --csv "$ROOT/Rapeseed/benchmark/quality/Rapeseed.csv" --method "GuidedDCNet"

echo "[queue] ALL QUALITY BENCHMARKS DONE"
