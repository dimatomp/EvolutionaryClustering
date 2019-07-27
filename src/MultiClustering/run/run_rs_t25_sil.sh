#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate


python3 ~/WORK/MultiClustering/RandomSearch.py iris.csv           42 sil 25 &
python3 ~/WORK/MultiClustering/RandomSearch.py glass.csv          42 sil 25  &
#python3 ~/WORK/MultiClustering/RandomSearch.py haberman.csv       42 sil 25  &
python3 ~/WORK/MultiClustering/RandomSearch.py wholesale.csv      42 sil 25  &
python3 ~/WORK/MultiClustering/RandomSearch.py yeast.csv          42 sil 25  &
python3 ~/WORK/MultiClustering/RandomSearch.py indiandiabests.csv 42 sil 25  &
#python3 ~/WORK/MultiClustering/RandomSearch.py krvskp.csv         42 sil 25  &


wait %1 %2 %3 %4 %5 #%6