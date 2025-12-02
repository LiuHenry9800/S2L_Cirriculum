#!/bin/bash

echo "Starting avg data selection"
python TRL/selection/data_selection.py --config TRL/configs/selection/avg.yml > avg_select.log 2>&1 


echo "Starting overall data selection"
python TRL/selection/data_selection.py --config TRL/configs/selection/overall.yml> ovr_select.log 2>&1 



echo "Starting 410M Avg Training"
python TRL/Train.py --config TRL/configs/410M_Avg.yml> avg_train.log 2>&1 

echo "Starting 410M Overall Training"
python TRL/Train.py --config TRL/configs/410M_Overall.yml > ovr_train.log 2>&1 





