#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate


python3 ~/WORK/MultiClustering/EXsmac.py iris.csv           42 ch 400  1>~/logs/ex-smac/iris.txt 2>~/logs/ex-smac/errors-iris.txt &
python3 ~/WORK/MultiClustering/EXsmac.py glass.csv          42 ch 400  1>~/logs/ex-smac/glass.txt 2>~/logs/ex-smac/errors-glass.txt &
python3 ~/WORK/MultiClustering/EXsmac.py haberman.csv       42 ch 400  1>~/logs/ex-smac/haberman.txt 2>~/logs/ex-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/EXsmac.py wholesale.csv      42 ch 400  1>~/logs/ex-smac/wholesale.txt 2>~/logs/ex-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/EXsmac.py yeast.csv          42 ch 400  1>~/logs/ex-smac/yeast.txt 2>~/logs/ex-smac/errors-yeast.txt &
python3 ~/WORK/MultiClustering/EXsmac.py indiandiabests.csv 42 ch 400  1>~/logs/ex-smac/indiandiabests.txt 2>~/logs/ex-smac/errors-indiandiabests.txt &
python3 ~/WORK/MultiClustering/EXsmac.py krvskp.csv         42 ch 400  1>~/logs/ex-smac/krvskp.txt 2>~/logs/ex-smac/errors-rkvskp.txt &

python3 ~/WORK/MultiClustering/RLsmac.py iris.csv           42 ch 1000 1 400 1>~/logs/rl-smac/iris.txt 2>~/logs/rl-smac/errors-iris.txt &
python3 ~/WORK/MultiClustering/RLsmac.py glass.csv          42 ch 1000 1 400 1>~/logs/rl-smac/glass.txt 2>~/logs/rl-smac/errors-glass.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv       42 ch 1000 1 400 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py wholesale.csv      42 ch 1000 1 400 1>~/logs/rl-smac/wholesale.txt 2>~/logs/rl-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RLsmac.py yeast.csv          42 ch 1000 1 400 1>~/logs/rl-smac/yeast.txt 2>~/logs/rl-smac/errors-yeast.txt &
python3 ~/WORK/MultiClustering/RLsmac.py indiandiabests.csv 42 ch 1000 1 400 1>~/logs/rl-smac/indiandiabests.txt 2>~/logs/rl-smac/errors-indiandiabests.txt &
python3 ~/WORK/MultiClustering/RLsmac.py krvskp.csv         42 ch 1000 1 400 1>~/logs/rl-smac/rkvskp.txt 2>~/logs/rl-smac/errors-krvskp.txt &


wait %1 %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 %12 %13 %14 # %15 %16 #%17 %18