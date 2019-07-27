#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 10 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 20 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 30 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 40 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 50 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 60 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 70 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 80 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 90 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 100 20 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &

python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 10 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 20 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 30 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 40 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 50 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 60 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 70 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 80 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 90 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RLsmac.py haberman.csv 42 calinski-harabasz 100 10 1>~/logs/rl-smac/haberman.txt 2>~/logs/rl-smac/errors-haberman.txt &

wait %1 %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 %12 %13 %14 %15 %16 %17 %18 %19 %20



