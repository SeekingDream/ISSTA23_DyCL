#!/bin/bash

CodeDir="compile_model/src_code/"
TaskList="0 1 2 3 4 5 6 7 8"
for task in $TaskList
do
  echo "$task"
  python compile_onnx.py --eval_id="$task"
  echo "$CodeDir""$task".py
done

for task in $TaskList
do
  python compile_tvm.py --eval_id="$task" --optimize=0
  python compile_tvm.py --eval_id="$task" --optimize=1
done