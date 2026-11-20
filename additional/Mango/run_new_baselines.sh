#!/bin/bash
set -e
cd "$(dirname "$0")"

python3 mango_spectraformer_classification_engine.py
python3 mango_resnet1d_classification_engine.py
python3 mango_spectraformer_regression_engine.py
python3 mango_resnet1d_regression_engine.py
