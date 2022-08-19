#!/bin/bash

for i in 0 1 2 3 4
do
  python3 translator.py stuff/griffiths_solver_vunedited/mawps_configs/config${i}OR.yaml
done
