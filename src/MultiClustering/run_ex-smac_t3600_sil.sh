#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

python3 ~/WORK/MultiClustering/EXsmac.py iris.csv           42 sil 3600
python3 ~/WORK/MultiClustering/EXsmac.py glass.csv          42 sil 3600
python3 ~/WORK/MultiClustering/EXsmac.py wholesale.csv      42 sil 3600
python3 ~/WORK/MultiClustering/EXsmac.py indiandiabests.csv 42 sil 3600
python3 ~/WORK/MultiClustering/EXsmac.py yeast.csv          42 sil 3600
python3 ~/WORK/MultiClustering/EXsmac.py krvskp.csv         42 sil 3600

wait %1 %2 %3 %4 %5 %6