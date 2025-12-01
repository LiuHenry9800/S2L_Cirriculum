#!/bin/bash

echo "Starting 70M Full Training"
python TRL/Train.py --config TRL/configs/70M_Full.yml

echo "Starting 410M Full Training"
python TRL/Train.py --config TRL/configs/410M_Full.yml
