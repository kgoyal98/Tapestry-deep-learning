#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/mnt/cherry/data/kgoyal/repos/graphsearch/tkipfgcn:/mnt/cherry/data/kgoyal/repos/graphsearch"

dir="/mnt/blossom/data/kgoyal/repos/Tapestry-deep-learning"
dir="."
num_queries=50
list=(5 6 7 10 13 17 20)
cpu=0
for i in ${list[@]}
do
  echo '---------------------------------------------------------------------------------'
  echo $i
  taskset -c $((30+$cpu))-$((30+$cpu)) python "$dir/train.py" --logfile "$dir/logs/sparsity_${i}.log"  --sparsity ${i}&
  let cpu=$cpu+1
done
