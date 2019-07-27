#!/bin/bash

source ~/WORK/environment/poni-multi-clustering/bin/activate


python3 ~/WORK/MultiClustering/ActiveStrategy.py haberman.csv   1 1>~/logs/active/haberman.txt 2>~/logs/active/errors-haberman.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py haberman.csv   11 1>~/logs/active/haberman.txt 2>~/logs/active/errors-haberman.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py haberman.csv   111 1>~/logs/active/haberman.txt 2>~/logs/active/errors-haberman.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py haberman.csv   211 1>~/logs/active/haberman.txt 2>~/logs/active/errors-haberman.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py haberman.csv   311 1>~/logs/active/haberman.txt 2>~/logs/active/errors-haberman.txt &


wait %1 %2 %3 %4 %5
