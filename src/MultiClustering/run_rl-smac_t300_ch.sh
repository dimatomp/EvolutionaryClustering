#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate


#python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv       42 ch 1000 1 300 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py iris.csv           42 ch 5000 1 300 1>~/logs/rl-smac/iris.txt 2>~/logs/rl-smac/errors-iris.txt &
python3 ~/WORK/MultiClustering/RLsmac.py glass.csv          42 ch 5000 1 300 1>~/logs/rl-smac/glass.txt 2>~/logs/rl-smac/errors-glass.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv      42 ch 5000 1 300 1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py yeast.csv          42 ch 5000 1 300 1>~/logs/rl-smac/yeast.txt 2>~/logs/rl-smac/errors-yeast.txt &
python3 ~/WORK/MultiClustering/RLsmac.py indiandiabests.csv 42 ch 5000 1 300 1>~/logs/rl-smac/indiandiabests.txt 2>~/logs/rl-smac/errors-indiandiabests.txt &
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 5000 1 300 1>~/logs/rl-smac/rkvskp.txt 2>~/logs/rl-smac/errors-krvskp.txt &


wait %1 %2 %3 %4 %5 %6 #%7