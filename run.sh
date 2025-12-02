#!/bin/bash

echo "Starting avg data selection"
python TRL/selection/data_selection.py --config TRL/configs/selection/avg.yml


echo "Starting overall data selection"
python TRL/selection/data_selection.py --config TRL/configs/selection/overall.yml



echo "Starting 410M Avg Training"
python TRL/Train.py --config TRL/configs/410M_Avg.yml

echo "Starting 410M Overall Training"
python TRL/Train.py --config TRL/configs/410M_Overall.yml





