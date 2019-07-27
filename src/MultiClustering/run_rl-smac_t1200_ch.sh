#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

python3 ~/WORK/MultiClustering/RLsmac.py iris.csv           42 ch 5000 1 1200
python3 ~/WORK/MultiClustering/RLsmac.py glass.csv          42 ch 5000 1 1200
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv      42 ch 5000 1 1200
python3 ~/WORK/MultiClustering/RLsmac.py indiandiabests.csv 42 ch 5000 1 1200
python3 ~/WORK/MultiClustering/RLsmac.py yeast.csv          42 ch 5000 1 1200
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 5000 1 1200

wait %1 %2 %3 %4 %5 %6