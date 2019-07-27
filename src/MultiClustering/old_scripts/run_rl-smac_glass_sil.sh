#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

#python3 ~/WORK/MultiClustering/RLsmac.py glass.csv 42 silhouette 800 1  1>~/logs/rl-smac/glass.txt 2>~/logs/rl-smac/errors-glass.txt &
python3 ~/WORK/MultiClustering/RLsmac.py glass.csv 42 silhouette 80 10  1>~/logs/rl-smac/glass.txt 2>~/logs/rl-smac/errors-glass.txt &
python3 ~/WORK/MultiClustering/RLsmac.py glass.csv 42 silhouette 40 20  1>~/logs/rl-smac/glass.txt 2>~/logs/rl-smac/errors-glass.txt &
python3 ~/WORK/MultiClustering/RLsmac.py glass.csv 42 silhouette 20 40  1>~/logs/rl-smac/glass.txt 2>~/logs/rl-smac/errors-glass.txt &
python3 ~/WORK/MultiClustering/RLsmac.py glass.csv 42 silhouette 10 80  1>~/logs/rl-smac/glass.txt 2>~/logs/rl-smac/errors-glass.txt &
python3 ~/WORK/MultiClustering/RLsmac.py glass.csv 42 silhouette 5 160  1>~/logs/rl-smac/glass.txt 2>~/logs/rl-smac/errors-glass.txt &

wait %1 %2 %3 %4 %5 #%6



