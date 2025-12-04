#!/bin/bash

echo "Starting S2L"
CUDA_VISIBLE_DEVICES=0 python TRL/Run.py --selection_config TRL/configs/selection/s2l.yml --train_config TRL/configs/410M_S2L.yml > s2l_pipeline.log 2>&1 &

echo "Starting avg"
CUDA_VISIBLE_DEVICES=1 python TRL/Run.py --selection_config TRL/configs/selection/avg.yml --train_config TRL/configs/410M_Avg.yml > avg_pipeline.log 2>&1 &

echo "Starting ovr"
CUDA_VISIBLE_DEVICES=2 python TRL/Run.py --selection_config TRL/configs/selection/overall.yml --train_config TRL/configs/410M_Overall.yml > overall_pipeline.log 2>&1 &

echo "Starting Instability"
CUDA_VISIBLE_DEVICES=3 python TRL/Run.py --selection_config TRL/configs/selection/instability.yml --train_config TRL/configs/410M_Instability.yml > instability_pipeline.log 2>&1 &

wait
echo "All pipelines complete"
