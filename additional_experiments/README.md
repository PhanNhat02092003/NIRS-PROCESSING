# additional_experiments

Benchmarks on external NIR datasets, kept apart from the Danang pesticide
pipeline (the FLAMENIR / OCEANFX work in the repository root and `reports/`).

    additional_experiments/
      Grainit/   Mango/   Rapeseed/   OSSL/
        prepare_<dataset>_dataset.py    build the dataset's ALL.csv from the raw download
        <dataset>_*_engine.py           training engines (classification, stage-2 regression, ...)
        benchmark/                      benchmark-format CSVs (not present for OSSL)
        checkpoint/  history/  data/    training outputs, same layout as in the root

Run any script from anywhere, e.g. `python3 additional_experiments/Grainit/grainit_classification_engine.py`.
Each script puts the repository root on `sys.path` (it reuses `model/`, `dataset/` and the shared
engines `food_classification.py`, `regression.py`, ...) and changes into its own
folder, so relative outputs (`history/`, `checkpoint/`, `data/`) are written next to the script.
Default raw-data locations are resolved against `../all-dataset` next to the repository.
