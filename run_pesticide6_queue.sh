#!/bin/bash
# Runs the 4 pesticide-6-substance jobs back-to-back on one GPU:
#   FLAMENIR XSpecMamba-joint -> FLAMENIR MT-SMARTNIR ->
#   OCEANFX  XSpecMamba-joint -> OCEANFX  MT-SMARTNIR
# OCEANFX needs BIN=8 (unbinned seq_len=2048 OOMs the CSSE encoder even at
# batch=64 -- measured directly before launching this queue).
set -e
cd /home/phannhat/NIRS-PROCESSING

export SUBSTANCES="Thiamethoxam,Permethrin,Azoxystrobin,Difenoconazole,Cypermethrin,Chlothianidin"
# Full FLAMENIR/OCEANFX (335k/380k rows) was far too slow per epoch (~4min
# with XSpecMamba's O(N^2) image precompute) -- subsample instead, stratified
# by food category, done before the expensive preprocessing pass.
# 20k measured R2 0.07-0.42 vs the old full-dataset single-substance
# baseline's 0.66-0.89 -- far too few present rows per substance (presence
# rate 7-38%) at that size. 100k is a deliberate speed/quality balance.
export SAMPLE_N=100000

echo "[queue] === FLAMENIR XSpecMamba-joint ==="
MACHINE=FLAMENIR python3 xspecmamba_joint_pesticide_engine.py

echo "[queue] === FLAMENIR MT-SMARTNIR ==="
MACHINE=FLAMENIR BATCH_SIZE=512 python3 multitask_mt_smartnir_engine.py

echo "[queue] === OCEANFX XSpecMamba-joint ==="
MACHINE=OCEANFX BIN=8 python3 xspecmamba_joint_pesticide_engine.py

echo "[queue] === OCEANFX MT-SMARTNIR ==="
MACHINE=OCEANFX BIN=8 BATCH_SIZE=128 python3 multitask_mt_smartnir_engine.py

echo "[queue] ALL PESTICIDE-6 JOBS DONE"
