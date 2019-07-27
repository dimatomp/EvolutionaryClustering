#!/usr/bin/env bash


sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t25_cop.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t50_cop.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t100_cop.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t200_cop.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t300_cop.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t400_cop.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t600_cop.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t1200_cop.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t2400_cop.sh


sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t25_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t50_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t100_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t200_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t300_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t400_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t600_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t1200_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t2400_sil.sh

sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t25_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t50_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t100_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t200_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t300_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t400_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t600_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t1200_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=6G -t 2000 run_rs_t2400_ch.sh

