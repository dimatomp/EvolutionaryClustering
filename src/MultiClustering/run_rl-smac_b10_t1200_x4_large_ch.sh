#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

python3 ~/WORK/MultiClustering/RLsmac.py yeast.csv          42 ch 5000 10 1200 rfrsls-smx-R-100
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 5000 10 1200 rfrsls-smx-R-100

python3 ~/WORK/MultiClustering/RLsmac.py yeast.csv          42 ch 5000 10 1200 rfrsls-ucb-SRU-100
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 5000 10 1200 rfrsls-ucb-SRU-100

python3 ~/WORK/MultiClustering/RLsmac.py yeast.csv          42 ch 5000 10 1200 rfrsls-ucb-SRSU-100
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 5000 10 1200 rfrsls-ucb-SRSU-100

python3 ~/WORK/MultiClustering/RLsmac.py yeast.csv          42 ch 5000 10 1200 rfrsls-smx-SRSU-100
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 5000 10 1200 rfrsls-smx-SRSU-100


wait %1 %2 %3 %4 %5 %6 %7 %8