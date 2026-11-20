# LensNIR (copied in, unmodified)

The 7 files here (`models.py`, `smartnir.py`, `kan.py`, `losses.py`,
`dataset_benchmark.py`, `train.py`, `evaluate.py`) are byte-identical copies
from `../../../LensNIR` (the original, untouched repo) -- just the subset
needed to train/evaluate on Grainit/Mango/Rapeseed, replacing the removed
MT-SMARTNIR method (`*_mt_smartnir_engine.py`/`*_mt_smartnir_benchmark.py`,
`model/mt_smartnir_model.py`'s Grainit/Mango/RapeseedCSSEModel) as the
multi-task comparison point for those 3 datasets. Not wired into this
repo's own `sys.path`/benchmark_utils.py conventions -- run it exactly as
documented in the original repo's README, from this folder.

`dataset_benchmark.py`'s `DEFAULT_DATA_ROOT` resolves relative to this
file's location (`<here>/../all-dataset`), which does not exist here --
the original repo relies on `DATA_ROOT` always being set explicitly rather
than the fallback. Either export it or pass `--data_root`:

```bash
export DATA_ROOT=/home/phannhat/all-dataset
# or: DATA_ROOT=/home/phannhat/all-dataset python3 train.py --dataset Grainit ...

python3 train.py --dataset Grainit \
  --d_model 128 --n_heads 4 --n_layers 4 --token_pool 2 --stem deep \
  --readout flat --readout_rank 32 --task_linear reg --cond_reg detach \
  --ema 0.999 --no_kan

python3 evaluate.py --dataset Grainit --checkpoint results/main/Grainit/fold1.pt <same training flags>
```

`--dataset` is `Grainit`, `Mango`, or `Rapeseed`; data is read straight from
the shared `/home/phannhat/all-dataset/{Zenodo-Grainit,Mendeley-Mango,
RechercheDataGouv-Rapeseed}/<Dataset>/ALL.csv` -- the same files this
repo's own `prepare_<dataset>_dataset.py` scripts already built, no
re-prep needed. Smoke-tested with `--epochs 2 --folds 1`: runs end to end.

The rest of the original repo (`baselines_classical.py`, `select_budget.py`,
`combine_st.py`, `summarize_benchmark.py`, `run_experiments.sh`, `scripts/`,
`paper/`) was left out as not needed just to run LensNIR here -- copy any
of it over the same way if needed later.
