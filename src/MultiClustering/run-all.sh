#!/usr/bin/env bash


sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t600_sil.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t400_sil.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t300_sil.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t200_sil.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t100_sil.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t50_sil.sh

sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t600_ch.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t400_ch.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t300_ch.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t200_ch.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t100_ch.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t50_ch.sh

sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t600_cop.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t400_cop.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t300_cop.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t200_cop.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t100_cop.sh
sbatch --cpus-per-task=14 --mail-type=END --mem=50G -t 2000 run_tl-smac_t50_cop.sh


sbatch --cpus-per-task=12 --mail-type=END --mem=40G -t 2000 run_tl-smac_t25_sil.sh
sbatch --cpus-per-task=12 --mail-type=END --mem=40G -t 2000 run_tl-smac_t25_ch.sh
sbatch --cpus-per-task=12 --mail-type=END --mem=40G -t 2000 run_tl-smac_t25_cop.sh

sbatch --cpus-per-task=6 --mail-type=END --mem=80G -t 2000 run_tl-smac_t1200_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=80G -t 2000 run_tl-smac_t1200_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=80G -t 2000 run_tl-smac_t1200_cop.sh
