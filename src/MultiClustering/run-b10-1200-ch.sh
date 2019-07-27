#!/usr/bin/env bash

sbatch --cpus-per-task=8 --mail-type=END --mem=10G -t 2000 run_rl-smac_b10_t1200_x4_large_ch.sh
sbatch --cpus-per-task=16 --mail-type=END --mem=16G -t 2000 run_rl-smac_b10_t1200_x4_small_ch.sh
