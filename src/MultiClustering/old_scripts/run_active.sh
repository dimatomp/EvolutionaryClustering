#!/bin/bash

source ~/WORK/environment/poni-multi-clustering/bin/activate


python3 ~/WORK/MultiClustering/ActiveStrategy.py car.csv        211 1>~/logs/active/car.txt 2>~/logs/active/errors-car.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py glass.csv      211 1>~/logs/active/glass.txt 2>~/logs/active/errors-glass.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py haberman.csv   211 1>~/logs/active/haberman.txt 2>~/logs/active/errors-haberman.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py ionosphere.csv 211 1>~/logs/active/ionosphere.txt 2>~/logs/active/errors-ionosphere.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py iris.csv       211 1>~/logs/active/iris.txt 2>~/logs/active/errors-iris.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py krvskp.csv     211 1>~/logs/active/krvskp.txt 2>~/logs/active/errors-krvskp.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py parcinson.csv  211 1>~/logs/active/parcinson.txt 2>~/logs/active/errors-parcinson.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py seeds.csv      211 1>~/logs/active/seeds.txt 2>~/logs/active/errors-seeds.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py semeion.csv    211 1>~/logs/active/semeion.txt 2>~/logs/active/errors-semeion.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py shuttle.csv    211 1>~/logs/active/shuttle.txt 2>~/logs/active/errors-shuttle.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py spect.csv      211 1>~/logs/active/spect.txt 2>~/logs/active/errors-spect.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py verebral.csv   211 1>~/logs/active/verebral.txt 2>~/logs/active/errors-verebral.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py waveform.csv   211 1>~/logs/active/waveform.txt 2>~/logs/active/errors-waveform.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py wholesale.csv  211 1>~/logs/active/wholesale.txt 2>~/logs/active/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py wine.csv       211 1>~/logs/active/wine.txt 2>~/logs/active/errors-wine.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py yeast.csv      211 1>~/logs/active/yeast.txt 2>~/logs/active/errors-yeast.txt &


python3 ~/WORK/MultiClustering/ActiveStrategy.py car.csv        311 1>~/logs/active/car.txt 2>~/logs/active/errors-car.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py glass.csv      311 1>~/logs/active/glass.txt 2>~/logs/active/errors-glass.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py haberman.csv   311 1>~/logs/active/haberman.txt 2>~/logs/active/errors-haberman.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py ionosphere.csv 311 1>~/logs/active/ionosphere.txt 2>~/logs/active/errors-ionosphere.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py iris.csv       311 1>~/logs/active/iris.txt 2>~/logs/active/errors-iris.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py krvskp.csv     311 1>~/logs/active/krvskp.txt 2>~/logs/active/errors-krvskp.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py parcinson.csv  311 1>~/logs/active/parcinson.txt 2>~/logs/active/errors-parcinson.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py seeds.csv      311 1>~/logs/active/seeds.txt 2>~/logs/active/errors-seeds.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py semeion.csv    311 1>~/logs/active/semeion.txt 2>~/logs/active/errors-semeion.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py shuttle.csv    311 1>~/logs/active/shuttle.txt 2>~/logs/active/errors-shuttle.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py spect.csv      311 1>~/logs/active/spect.txt 2>~/logs/active/errors-spect.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py verebral.csv   311 1>~/logs/active/verebral.txt 2>~/logs/active/errors-verebral.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py waveform.csv   311 1>~/logs/active/waveform.txt 2>~/logs/active/errors-waveform.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py wholesale.csv  311 1>~/logs/active/wholesale.txt 2>~/logs/active/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py wine.csv       311 1>~/logs/active/wine.txt 2>~/logs/active/errors-wine.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py yeast.csv      311 1>~/logs/active/yeast.txt 2>~/logs/active/errors-yeast.txt &



wait %1 %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 %12 %13 %14 %15 %16 %17 %18 %19 %20 %21 %22 %23 %24 %25 %26 %27 %28 %29 %30 %31 %32