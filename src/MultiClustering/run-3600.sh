#!/usr/bin/env bash

sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t3600_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t3600_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t3600_cop.sh

sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t3600_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t3600_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t3600_cop.sh

