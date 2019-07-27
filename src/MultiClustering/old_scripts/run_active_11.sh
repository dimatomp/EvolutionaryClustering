#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate


#python3 ~/WORK/MultiClustering/ActiveStrategy.py glass.csv      11 1>~/logs/active/glass.txt 2>~/logs/active/errors-glass.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py car.csv            11 1>~/logs/active/car.txt 2>~/logs/active/errors-car.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py haberman.csv       11 1>~/logs/active/haberman.txt 2>~/logs/active/errors-haberman.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py ionosphere.csv     11 1>~/logs/active/ionosphere.txt 2>~/logs/active/errors-ionosphere.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py iris.csv           11 1>~/logs/active/iris.txt 2>~/logs/active/errors-iris.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py parcinson.csv      11 1>~/logs/active/parcinson.txt 2>~/logs/active/errors-parcinson.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py seeds.csv          11 1>~/logs/active/seeds.txt 2>~/logs/active/errors-seeds.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py spect.csv          11 1>~/logs/active/spect.txt 2>~/logs/active/errors-spect.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py verebral.csv       11 1>~/logs/active/verebral.txt 2>~/logs/active/errors-verebral.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py wholesale.csv      11 1>~/logs/active/wholesale.txt 2>~/logs/active/errors-wholesale.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py wine.csv           11 1>~/logs/active/wine.txt 2>~/logs/active/errors-wine.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py yeast.csv          11 1>~/logs/active/yeast.txt 2>~/logs/active/errors-yeast.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py backup.csv         11 1>~/logs/active/backup.txt         2>~/logs/active/errors-backup.txt         &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py balance.csv        11 1>~/logs/active/balance.txt        2>~/logs/active/errors-balance.txt        &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py banknote.csv       11 1>~/logs/active/banknote.txt       2>~/logs/active/errors-banknote.txt       &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py indiandiabests.csv 11 1>~/logs/active/indiandiabests.txt 2>~/logs/active/errors-indiandiabests.txt &
#python3 ~/WORK/MultiClustering/ActiveStrategy.py tae.csv            11 1>~/logs/active/tae.txt            2>~/logs/active/errors-tae.txt            &
#
#
#wait %1 %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 %12 %13 %14 %15 %16 %17

python3 ~/WORK/MultiClustering/ActiveStrategy.py ecoli.csv         11 1>~/logs/active/ecoli.txt       2>~/logs/active/errors-ecoli.txt       &
python3 ~/WORK/MultiClustering/ActiveStrategy.py flags.csv         11 1>~/logs/active/flags.txt       2>~/logs/active/errors-flags.txt       &
python3 ~/WORK/MultiClustering/ActiveStrategy.py forestfires.csv   11 1>~/logs/active/forestfires.txt 2>~/logs/active/errors-forestfires.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py leaf.csv          11 1>~/logs/active/leaf.txt        2>~/logs/active/errors-leaf.txt        &
wait %1 %2 %3 %4

