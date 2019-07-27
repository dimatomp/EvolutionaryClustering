#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

#python3 ~/WORK/MultiClustering/RLsmac.py iris.csv 42 silhouette 800 1  1>~/logs/rl-smac/iris.txt 2>~/logs/rl-smac/errors-iris.txt &
#python3 ~/WORK/MultiClustering/RLsmac.py iris.csv 42 silhouette 80 10  1>~/logs/rl-smac/iris.txt 2>~/logs/rl-smac/errors-iris.txt &
#python3 ~/WORK/MultiClustering/RLsmac.py iris.csv 42 silhouette 40 20  1>~/logs/rl-smac/iris.txt 2>~/logs/rl-smac/errors-iris.txt &
python3 ~/WORK/MultiClustering/RLsmac.py iris.csv 42 silhouette 20 40  1>~/logs/rl-smac/iris.txt 2>~/logs/rl-smac/errors-iris.txt &
#python3 ~/WORK/MultiClustering/RLsmac.py iris.csv 42 silhouette 10 80  1>~/logs/rl-smac/iris.txt 2>~/logs/rl-smac/errors-iris.txt &
python3 ~/WORK/MultiClustering/RLsmac.py iris.csv 42 silhouette 5 160  1>~/logs/rl-smac/iris.txt 2>~/logs/rl-smac/errors-iris.txt &


wait %1 %2 #%3 %4 %5 %6 #%7 %8 %9 %10 %11 %12 %13 %14 %15 %16



