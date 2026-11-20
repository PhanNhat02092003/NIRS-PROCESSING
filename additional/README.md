# additional

Benchmarks on external NIR datasets, kept apart from the Danang pesticide
pipeline (the FLAMENIR / OCEANFX work in `../main/` and `../main/reports/`).

    additional/
      Grainit/   Mango/   Rapeseed/
        prepare_<dataset>_dataset.py    build the dataset's ALL.csv from the raw download
        <dataset>_*_engine.py           training engines (classification, stage-2 regression, ...)
        benchmark/                      benchmark-format CSVs
        checkpoint/  history/  data/    training outputs, same layout as in main/
      model/    dataset/                 model/dataset modules used only here (EBAR, NirMACNet,
                                          XSpecMamba, PLS, ResNet1D, SpectraFormer, the per-dataset
                                          quality/regression Dataset classes, ...)
      baseline-information/              reference papers/templates for the baseline architectures
      LensNIR/                           LensNIR (replaces MT-SMARTNIR as the multi-task comparison
                                          point for Grainit/Mango/Rapeseed) -- unmodified copy of
                                          the relevant files from ../../../LensNIR, see its README

Run any script from anywhere, e.g. `python3 additional/Grainit/grainit_classification_engine.py`.
Each script puts `../main/` AND `../` (this `additional/` directory) on `sys.path`, so `import
model.x` / `import dataset.x` transparently merge `main/model/` + `additional/model/` (and the
`dataset/` equivalents) into one namespace package -- shared modules (classification_model,
guideddcnet_model, classification_dataset, preprocessing) live in `main/`, everything used only
by these benchmarks lives in `additional/model/` and `additional/dataset/`. Scripts also reuse the
shared training loops in `main/food_classification.py` (`FocalLoss`, `train`, `pretrain`,
`train_diffusion`) and change into their own folder, so relative outputs (`history/`,
`checkpoint/`, `data/`) are written next to the script.
Default raw-data locations are resolved against `../../all-dataset` next to the repository.
