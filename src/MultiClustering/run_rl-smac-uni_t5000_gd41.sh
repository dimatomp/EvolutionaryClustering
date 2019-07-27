#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

#python3 ~/WORK/MultiClustering/RLsmac.py iris.csv           42 cop 5000 1 5000 rfrsls-uni
#python3 ~/WORK/MultiClustering/RLsmac.py glass.csv          42 cop 5000 1 5000 rfrsls-uni
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv      42 gd41 5000 1 10000 rfrsls-uni
python3 ~/WORK/MultiClustering/RLsmac.py indiandiabests.csv 42 gd41 5000 1 10000 rfrsls-uni
python3 ~/WORK/MultiClustering/RLsmac.py yeast.csv          42 gd41 5000 1 10000 rfrsls-uni
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 gd41 5000 1 10000 rfrsls-uni

wait %1 %2 %3 %4 #%5 %6