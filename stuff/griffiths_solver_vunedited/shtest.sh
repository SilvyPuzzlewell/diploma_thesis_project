#!/bin/bash

for i in 0 1 2 3 4
do
  echo "./mawps_configs/config$i.yaml"
  echo "./mawps_configs/config${i}CZ.yaml"
  echo "./mawps_configs/config${i}OR.yaml"
done
