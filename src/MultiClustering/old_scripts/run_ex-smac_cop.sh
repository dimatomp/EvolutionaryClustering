#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate


python3 ~/WORK/MultiClustering/EXsmac.py iris.csv           42 cop 200  1>~/logs/ex-smac/iris.txt 2>~/logs/ex-smac/errors-iris.txt &
python3 ~/WORK/MultiClustering/EXsmac.py glass.csv          42 cop 200  1>~/logs/ex-smac/glass.txt 2>~/logs/ex-smac/errors-glass.txt &
python3 ~/WORK/MultiClustering/EXsmac.py haberman.csv       42 cop 200  1>~/logs/ex-smac/haberman.txt 2>~/logs/ex-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/EXsmac.py wholesale.csv      42 cop 200  1>~/logs/ex-smac/wholesale.txt 2>~/logs/ex-smac/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/EXsmac.py yeast.csv          42 cop 200  1>~/logs/ex-smac/yeast.txt 2>~/logs/ex-smac/errors-yeast.txt &
python3 ~/WORK/MultiClustering/EXsmac.py indiandiabests.csv 42 cop 200  1>~/logs/ex-smac/indiandiabests.txt 2>~/logs/ex-smac/errors-indiandiabests.txt &
#python3 ~/WORK/MultiClustering/EXsmac.py krvskp.csv         42 cop 200  1>~/logs/ex-smac/krvskp.txt 2>~/logs/ex-smac/errors-krvskp.txt &


wait %1 %2 %3 %4 %5 %6 # %7 #%8 %9 %10 %11 %12 %13 %14 %15 %16



