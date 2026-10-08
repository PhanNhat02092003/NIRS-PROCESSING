#!/bin/bash
# MT-SMARTNIR only (FLAMENIR -> OCEANFX), same SUBSTANCES/SAMPLE_N as the
# pesticide-6 queue. XSpecMamba-joint was stopped early by request (2/5
# FLAMENIR folds done, enough for a comparison point).
set -e
cd /home/phannhat/NIRS-PROCESSING

export SUBSTANCES="Thiamethoxam,Permethrin,Azoxystrobin,Difenoconazole,Cypermethrin,Chlothianidin"
export SAMPLE_N=100000

echo "[queue] === FLAMENIR MT-SMARTNIR ==="
MACHINE=FLAMENIR BATCH_SIZE=512 python3 multitask_mt_smartnir_engine.py

echo "[queue] === OCEANFX MT-SMARTNIR ==="
MACHINE=OCEANFX BIN=8 BATCH_SIZE=128 python3 multitask_mt_smartnir_engine.py

echo "[queue] ALL MT-SMARTNIR JOBS DONE"
