#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

#python3 ~/WORK/MultiClustering/ProfileMetrics.py  calinski-harabasz  2000 &
python3 ~/WORK/MultiClustering/ProfileMetrics.py  silhouette         10 &
python3 ~/WORK/MultiClustering/ProfileMetrics.py  cs                 10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  score-function     10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  sym                10 &
python3 ~/WORK/MultiClustering/ProfileMetrics.py  gd41               10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  gd43               10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  dunn               10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  davies-bouldin     10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  gd31               10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  gd51               10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  gd33               10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  gd53               10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  db-star            10 &
python3 ~/WORK/MultiClustering/ProfileMetrics.py  cop                10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  sv                 10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  os                 10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  sym-davies-bouldin 10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  s-dbw              10 &
#python3 ~/WORK/MultiClustering/ProfileMetrics.py  c-ind              10 &

wait %1 %2 %3 %4 #%5 %6 %7 %8 %9 %10 %11 %12 %13 %14 %15 %16 %17 %18 %19
