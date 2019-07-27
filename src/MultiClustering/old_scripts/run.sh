#!/bin/bash

echo activate environment

source ~/WORK/environment/poni-multi-clustering/bin/activate

echo start experiments


# Run both processes and wain for them.
python3 ~/WORK/MultiClustering/ActiveStrategy.py car.csv        all 1>~/logs/active/car.txt 2>~/logs/active/errors-car.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py glass.csv      all 1>~/logs/active/glass.txt 2>~/logs/active/errors-glass.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py haberman.csv   all 1>~/logs/active/haberman.txt 2>~/logs/active/errors-haberman.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py ionosphere.csv all 1>~/logs/active/ionosphere.txt 2>~/logs/active/errors-ionosphere.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py iris.csv       all 1>~/logs/active/iris.txt 2>~/logs/active/errors-iris.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py krvskp.csv     all 1>~/logs/active/krvskp.txt 2>~/logs/active/errors-krvskp.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py parcinson.csv  all 1>~/logs/active/parcinson.txt 2>~/logs/active/errors-parcinson.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py seeds.csv      all 1>~/logs/active/seeds.txt 2>~/logs/active/errors-seeds.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py semeion.csv    all 1>~/logs/active/semeion.txt 2>~/logs/active/errors-semeion.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py shuttle.csv    all 1>~/logs/active/shuttle.txt 2>~/logs/active/errors-shuttle.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py spect.csv      all 1>~/logs/active/spect.txt 2>~/logs/active/errors-spect.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py verebral.csv   all 1>~/logs/active/verebral.txt 2>~/logs/active/errors-verebral.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py waveform.csv   all 1>~/logs/active/waveform.txt 2>~/logs/active/errors-waveform.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py wholesale.csv  all 1>~/logs/active/wholesale.txt 2>~/logs/active/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py wine.csv       all 1>~/logs/active/wine.txt 2>~/logs/active/errors-wine.txt &
python3 ~/WORK/MultiClustering/ActiveStrategy.py yeast.csv      all 1>~/logs/active/yeast.txt 2>~/logs/active/errors-yeast.txt &

echo "Experiments started on pony" | mail -s "Experiment started" sslavian812@gmail.com

wait %1 %2 %3 %4 %5 %6 %7 %8 %9 %10 %11 %12 %13 %14 %15 %16

echo experiments completed

# Pack results together for future downloading, cleanup.
#DATE=`date '+%Y-%m-%d_%H:%M:%S'`
#mkdir ~/archive/${DATE}
#
#cp ~/WORK/MultiClustering/result ~/archive/${DATE}/result

# mv ~/logs/exh ~/archive/${DATE}/logs-exh
# mv ~/logs/active ~/archive/${DATE}/logs-active
# mv ~/logs/errors* ~/archive/${DATE}/

# TODO(shalamov): delete result folder content only after checking that it is successfully stored somewhere else

# Remove smac-generated temporal files.
sh ~/WORK/MultiClustering/clean


echo "To download results to local machine use: \n scp vshalamov@genome.ifmo.ru:WORK/MultiClustering/result/* ." | \
    mail -s "Experiment finished" sslavian812@gmail.com



