#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

#python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 3000 1 5000 rfrsls-smx-R-100
#python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 3000 1 5000 rfrsls-ucb-SRU-100
#python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 3000 1 5000 rfrsls-ucb-SRSU-100
#python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 3000 1 5000 rfrsls-smx-SRSU-100
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 5000 1 1200 rfrsls-uni
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 5000 1 1200 rfrsls-uni

wait %1 %2 # %3 %4