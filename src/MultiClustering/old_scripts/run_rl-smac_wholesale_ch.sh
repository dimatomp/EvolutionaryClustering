#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 calinski-harabasz 800 1  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 calinski-harabasz 80 10  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 calinski-harabasz 40 20  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 calinski-harabasz 20 40  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 calinski-harabasz 10 80  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 calinski-harabasz 5 160  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &

wait %1 %2 %3 %4 %5 %6



