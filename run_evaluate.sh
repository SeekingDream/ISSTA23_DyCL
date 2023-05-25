#!/bin/bash


TaskList="0 1 2 3 4 5 6 7 8"
for task in $TaskList
do
  echo "$task"
  python evaluate_onnx.py --eval_id="$task" --optimize=0
  python evaluate_onnx.py --eval_id="$task" --optimize=1
  python evaluate_tvm.py --eval_id="$task" --optimize=0
  python evaluate_tvm.py --eval_id="$task" --optimize=1
done

