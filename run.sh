#!/bin/bash

echo "Starting avg "
python TRL/Run.py --selection_config TRL/configs/selection/avg.yml --train_config TRL/configs/410M_Avg.yml > avg_pipeline.log 2>&1
echo "avg complete"

echo "Starting ovr"
python TRL/Run.py --selection_config TRL/configs/selection/overall.yml --train_config TRL/configs/410M_Overall.yml > overall_pipeline.log 2>&1
echo "ovr complete"

echo "Starting Instability"
python TRL/Run.py --selection_config TRL/configs/selection/instability.yml --train_config TRL/configs/410M_Instability.yml > instability_pipeline.log 2>&1
echo "Instability complete!"





