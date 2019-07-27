#!/bin/bash

cd ~/WORK/MultiClustering

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


python3 ~/WORK/MultiClustering/ExhaustiveSearch.py car.csv        all 1>~/logs/exh/car.txt 2>~/logs/exh/errors-car.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py glass.csv      all 1>~/logs/exh/glass.txt 2>~/logs/exh/errors-glass.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py haberman.csv   all 1>~/logs/exh/haberman.txt 2>~/logs/exh/errors-haberman.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py ionosphere.csv all 1>~/logs/exh/ionosphere.txt 2>~/logs/exh/errors-ionosphere.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py iris.csv       all 1>~/logs/exh/iris.txt 2>~/logs/exh/errors-iris.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py krvskp.csv     all 1>~/logs/exh/krvskp.txt 2>~/logs/exh/errors-krvskp.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py parcinson.csv  all 1>~/logs/exh/parcinson.txt 2>~/logs/exh/errors-parcinson.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py seeds.csv      all 1>~/logs/exh/seeds.txt 2>~/logs/exh/errors-seeds.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py semeion.csv    all 1>~/logs/exh/semeion.txt 2>~/logs/exh/errors-semeion.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py shuttle.csv    all 1>~/logs/exh/shuttle.txt 2>~/logs/exh/errors-shuttle.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py spect.csv      all 1>~/logs/exh/spect.txt 2>~/logs/exh/errors-spect.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py verebral.csv   all 1>~/logs/exh/verebral.txt 2>~/logs/exh/errors-verebral.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py waveform.csv   all 1>~/logs/exh/waveform.txt 2>~/logs/exh/errors-waveform.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py wholesale.csv  all 1>~/logs/exh/wholesale.txt 2>~/logs/exh/errors-wholesale.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py wine.csv       all 1>~/logs/exh/wine.txt 2>~/logs/exh/errors-wine.txt &
python3 ~/WORK/MultiClustering/ExhaustiveSearch.py yeast.csv      all 1>~/logs/exh/yeast.txt 2>~/logs/exh/errors-yeast.txt &

