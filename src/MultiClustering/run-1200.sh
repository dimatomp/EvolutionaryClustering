#!/usr/bin/env bash

#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_ch.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_sil.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_cop.sh

sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac-uni_t1200_ch.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac-uni_t1200_sil.sh
sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac-uni_t1200_cop.sh
#sbatch --cpus-per-task=2 --mail-type=END --mem=9G -t 2000 run_rl-smac-uni_t5000_cop.sh
#sbatch --cpus-per-task=2 --mail-type=END --mem=9G -t 2000 run_rl-smac-uni_t5000_os.sh
#sbatch --cpus-per-task=2 --mail-type=END --mem=9G -t 2000 run_rl-smac-uni_t5000_gd41.sh

#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_ch.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_sil.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_cop.sh


sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac-uni_t1200_sym.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac-uni_t1200_os.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac-uni_t1200_gd41.sh

#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_sym.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_os.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_gd41.sh

#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_gd43.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_gd33.sh

#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_sym.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_os.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_gd41.sh


#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_gd43.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_gd33.sh


#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_gd51.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_gd53.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_gd31.sh

#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_gd51.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_gd53.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_gd31.sh



#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_dunn.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_db.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_cs.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_db-star.sh

#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_dunn.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_db.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_cs.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_db-star.sh



#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_sf.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_sym-db.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_s-dbw.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_ex-smac_t1200_c-ind.sh
#
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_sf.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_sym-db.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_s-dbw.sh
#sbatch --cpus-per-task=6 --mail-type=END --mem=9G -t 2000 run_rl-smac_t1200_c-ind.sh

