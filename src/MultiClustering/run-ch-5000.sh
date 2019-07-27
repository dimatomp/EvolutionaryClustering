#!/usr/bin/env bash

sbatch --cpus-per-task=2 --mail-type=END --mem=10G -t 2000 run_rl-smac_t5000_ch.sh

