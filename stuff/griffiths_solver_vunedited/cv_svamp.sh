#!/bin/bash

for i in 0 1 2 3 4
do
  python3 translator.py ./svamp_configs/config$i.yaml
  python3 translator.py ./svamp_configs/config${i}CZ.yaml
  python3 translator.py ./svamp_configs/config${i}OR.yaml
done
