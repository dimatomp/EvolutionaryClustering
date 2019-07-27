#!/bin/bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

python3 ~/WORK/MultiClustering/RandomSearch.py car.csv        1 all 1>~/logs/exh/car.txt 2>~/logs/exh/errors-car.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py glass.csv      1 all 1>~/logs/exh/glass.txt 2>~/logs/exh/errors-glass.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py haberman.csv   1 all 1>~/logs/exh/haberman.txt 2>~/logs/exh/errors-haberman.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py ionosphere.csv 1 all 1>~/logs/exh/ionosphere.txt 2>~/logs/exh/errors-ionosphere.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py iris.csv       1 all 1>~/logs/exh/iris.txt 2>~/logs/exh/errors-iris.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py krvskp.csv     1 all 1>~/logs/exh/krvskp.txt 2>~/logs/exh/errors-krvskp.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py parcinson.csv  1 all 1>~/logs/exh/parcinson.txt 2>~/logs/exh/errors-parcinson.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py seeds.csv      1 all 1>~/logs/exh/seeds.txt 2>~/logs/exh/errors-seeds.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py semeion.csv    1 all 1>~/logs/exh/semeion.txt 2>~/logs/exh/errors-semeion.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py shuttle.csv    1 all 1>~/logs/exh/shuttle.txt 2>~/logs/exh/errors-shuttle.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py spect.csv      1 all 1>~/logs/exh/spect.txt 2>~/logs/exh/errors-spect.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py verebral.csv   1 all 1>~/logs/exh/verebral.txt 2>~/logs/exh/errors-verebral.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py waveform.csv   1 all 1>~/logs/exh/waveform.txt 2>~/logs/exh/errors-waveform.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py wholesale.csv  1 all 1>~/logs/exh/wholesale.txt 2>~/logs/exh/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py wine.csv       1 all 1>~/logs/exh/wine.txt 2>~/logs/exh/errors-wine.txt &
python3 ~/WORK/MultiClustering/RandomSearch.py yeast.csv      1 all 1>~/logs/exh/yeast.txt 2>~/logs/exh/errors-yeast.txt &

wait %1 %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 %12 %13 %14 %15 %16