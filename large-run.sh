#!/bin/bash

# Train 70M proxy model on 210K samples
echo "Training 70M model"
CUDA_VISIBLE_DEVICES=0 python TRL/Train.py --config TRL/large_configs/70M_Full.yml > 70m_pipeline.log 2>&1

# Collect trajectories
echo "getting trajectories"
CUDA_VISIBLE_DEVICES=0 python TRL/selection/get_trajectories.py --checkpoint_dir ./large_results/pythia-70M-full --dataset_name TIGER-Lab/MathInstruct --n_samples 240000 > traj_pipeline.log 2>&1

# Run pipelines
echo "Starting Full"
CUDA_VISIBLE_DEVICES=1 python TRL/Train.py --config TRL/large_configs/410M_Full.yml > large_full.log 2>&1 &

echo "Starting S2L"
CUDA_VISIBLE_DEVICES=0 python TRL/Run.py --selection_config TRL/large_configs/selection/s2l.yml --train_config TRL/large_configs/410M_S2L.yml > large_s2l.log 2>&1 &

echo "Starting avg"
CUDA_VISIBLE_DEVICES=1 python TRL/Run.py --selection_config TRL/large_configs/selection/avg.yml --train_config TRL/large_configs/410M_Avg.yml > large_avg.log 2>&1 &

echo "Starting ovr"
CUDA_VISIBLE_DEVICES=2 python TRL/Run.py --selection_config TRL/large_configs/selection/overall.yml --train_config TRL/large_configs/410M_Overall.yml > large_overall.log 2>&1 &

echo "Starting Instability"
CUDA_VISIBLE_DEVICES=3 python TRL/Run.py --selection_config TRL/large_configs/selection/instability.yml --train_config TRL/large_configs/410M_Instability.yml > large_instability.log 2>&1 &

wait
echo "All pipelines complete"
