#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 cop 800 1  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 cop 80 10  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 cop 40 20  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 cop 20 40  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 cop 10 80  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv 42 cop 5 160  1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &

wait %1 %2 %3 %4 %5 %6



