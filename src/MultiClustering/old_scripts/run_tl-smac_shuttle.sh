#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate



#python3 ~/WORK/MultiClustering/EXsmac.py shuttle.csv        42 ch  900  1>~/logs/ex-smac/shuttle.txt 2>~/logs/ex-smac/errors-shuttle.txt &
#python3 ~/WORK/MultiClustering/EXsmac.py shuttle.csv        42 sil 900  1>~/logs/ex-smac/shuttle.txt 2>~/logs/ex-smac/errors-shuttle.txt &
#python3 ~/WORK/MultiClustering/EXsmac.py shuttle.csv        42 cop 900  1>~/logs/ex-smac/shuttle.txt 2>~/logs/ex-smac/errors-shuttle.txt &
#
#python3 ~/WORK/MultiClustering/EXsmac.py shuttle.csv        42 ch  600  1>~/logs/ex-smac/shuttle.txt 2>~/logs/ex-smac/errors-shuttle.txt &
#python3 ~/WORK/MultiClustering/EXsmac.py shuttle.csv        42 sil 600  1>~/logs/ex-smac/shuttle.txt 2>~/logs/ex-smac/errors-shuttle.txt &
#python3 ~/WORK/MultiClustering/EXsmac.py shuttle.csv        42 cop 600  1>~/logs/ex-smac/shuttle.txt 2>~/logs/ex-smac/errors-shuttle.txt &
#
#python3 ~/WORK/MultiClustering/EXsmac.py shuttle.csv        42 ch  300  1>~/logs/ex-smac/shuttle.txt 2>~/logs/ex-smac/errors-shuttle.txt &
#python3 ~/WORK/MultiClustering/EXsmac.py shuttle.csv        42 sil 300  1>~/logs/ex-smac/shuttle.txt 2>~/logs/ex-smac/errors-shuttle.txt &
#python3 ~/WORK/MultiClustering/EXsmac.py shuttle.csv        42 cop 300  1>~/logs/ex-smac/shuttle.txt 2>~/logs/ex-smac/errors-shuttle.txt &
#

python3 ~/WORK/MultiClustering/RLsmac.py shuttle.csv        42 ch  1000 1 1200 1>~/logs/rl-smac/shuttle.txt 2>~/logs/rl-smac/errors-shuttle.txt &
python3 ~/WORK/MultiClustering/RLsmac.py shuttle.csv        42 sil 1000 1 1200 1>~/logs/rl-smac/shuttle.txt 2>~/logs/rl-smac/errors-shuttle.txt &
python3 ~/WORK/MultiClustering/RLsmac.py shuttle.csv        42 cop 1000 1 1200 1>~/logs/rl-smac/shuttle.txt 2>~/logs/rl-smac/errors-shuttle.txt &


wait %1 %2 %3 #%4 %5 %6 %7 %8 %9 %10 %11 %12 # %13 %14 %15 %16 #%17 %18