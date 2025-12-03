import argparse
import yaml
import os
import sys
import json
import glob
import numpy as np
import wandb

from TRL.selection.get_trajectories import compute_losses
from TRL.selection.data_selection import * # wildcard import, change later
from TRL.Train import TrainConfig, train_model

# get the loss trajectories
def get_trajectories(checkpoint_path, dataset_name, n_samples, run_name):
    for checkpoint in checkpoint_path:
        # get the losses for each checkpoint
        print(f"getting losses for {checkpoint}")
        try:
            compute_losses(os.path.join(checkpoint_path, checkpoint), dataset_name, n_samples, run_name)
        except Exception as e:
            print(f"{e=}")


def data_selection(losses_dir=None, dataset_name=None, n_samples=None, algorithm=None,
                   dataset_cutoff=None, num_epochs=None, output_file=None, config_file=None):
    """
    Args:
        losses_dir (str): Directory containing checkpoint loss files
        dataset_name (str): HuggingFace dataset name or local path
        n_samples (int): Number of samples to select (e.g., 60000)
        algorithm (str): Selection algorithm
        dataset_cutoff (int): Maximum samples to load from dataset (default: 120000)
        num_epochs (int): Number of difficulty levels for curriculum algorithms (default: 3)
                         For curriculum algorithms, total samples = n_samples * num_epochs
        output_file (str): Path to save selected data JSON (e.g., "./results/data/70M-S2L.json")
        config_file (str): Optional path to YAML config file. If provided, loads params from file and overrides with any passed args

    Returns:
        selected_indices (np.ndarray): Array of selected sample indices
    """
    config = SelectionConfig(config_file=config_file)

    # Override with any passed parameters
    if losses_dir is not None:
        config.losses_dir = losses_dir
    if dataset_name is not None:
        config.dataset_name = dataset_name
    if dataset_cutoff is not None:
        config.dataset_cutoff = dataset_cutoff
    if n_samples is not None:
        config.n_samples = n_samples
    if algorithm is not None:
        config.algorithm = algorithm
    if num_epochs is not None:
        config.num_epochs = num_epochs
    if output_file is not None:
        config.output_file = output_file

    os.makedirs(os.path.dirname(config.output_file), exist_ok=True)

    selected_indices = select_training_data(config)
    print(f"Data selection completed: {len(selected_indices)} samples selected")
    print(f"Selected data saved to: {config.output_file}")

    # sort of a useless return
    return selected_indices


def train_70m(config_file=None, model_path=None, dataset_name=None, n_samples=None,
              output_dir=None, num_train_epochs=None, save_steps=None):
    """
    Train 70M model

    Args:
        config_file (str): Optional path to YAML config file
        model_path (str): HuggingFace model path (default: "EleutherAI/pythia-70m-deduped")
        dataset_name (str): Dataset name or path (default: "TIGER-Lab/MathInstruct")
        n_samples (int): Number of samples to train on (default: -1 for all)
        output_dir (str): Output directory (default: "./pythia-70m-math")
        num_train_epochs (int): Number of training epochs (default: 3)
        save_steps (int): Save checkpoint every N steps (default: 3750)

    Returns:
        output_dir (str): Path to trained model
    """
    config = TrainConfig(config_file=config_file)

    # Override with any passed parameters
    if model_path is not None:
        config.model_path = model_path
    if dataset_name is not None:
        config.dataset_name = dataset_name
    if n_samples is not None:
        config.n_samples = n_samples
    if output_dir is not None:
        config.output_dir = output_dir
    if num_train_epochs is not None:
        config.num_train_epochs = num_train_epochs
    if save_steps is not None:
        config.save_steps = save_steps

    train_model(config)
    return config.output_dir


def train_410m(config_file=None, model_path=None, dataset_name=None, n_samples=None,
               output_dir=None, num_train_epochs=None, save_steps=None):
    """
    Args:
        config_file (str): Optional path to YAML config file
        model_path (str): HuggingFace model path (default: "EleutherAI/pythia-410m-deduped")
        dataset_name (str): Dataset name or path (default: "TIGER-Lab/MathInstruct")
        n_samples (int): Number of samples to train on (default: -1 for all)
        output_dir (str): Output directory (default: "./pythia-410m-math")
        num_train_epochs (int): Number of training epochs (default: 3)
        save_steps (int): Save checkpoint every N steps (default: 3750)

    Returns:
        output_dir (str): Path to trained model
    """
    config = TrainConfig(config_file=config_file)

    # Override with any passed parameters
    if model_path is not None:
        config.model_path = model_path
    if dataset_name is not None:
        config.dataset_name = dataset_name
    if n_samples is not None:
        config.n_samples = n_samples
    if output_dir is not None:
        config.output_dir = output_dir
    if num_train_epochs is not None:
        config.num_train_epochs = num_train_epochs
    if save_steps is not None:
        config.save_steps = save_steps

    train_model(config)
    return config.output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S2L Curriculum Learning Pipeline")

    # Main pipeline flags
    parser.add_argument('--train-70m', action='store_true',
                       help='Train 70M model')
    parser.add_argument('--get-trajectories', action='store_true',
                       help='Generate loss trajectories from 70M checkpoints')
    parser.add_argument('--config-70m', type=str, default=None,
                       help='Path to YAML config file for 70M training')
    parser.add_argument('--config-selection', type=str, default=None,
                       help='Path to YAML config file for data selection')
    parser.add_argument('--config-410m', type=str, default=None,
                       help='Path to YAML config file for 410M training')
    parser.add_argument('--train-410m', action='store_true',
                       help='Train 410M model')

    args = parser.parse_args()

    if args.train_70m:
        if args.config_70m is not None:
            # Use config file
            output_dir = train_70m(config_file=args.config_70m)
        else:
            # Use default dict
            train_70m_params = {
                "model_path": "EleutherAI/pythia-70m-deduped",
                "dataset_name": "TIGER-Lab/MathInstruct",
                "n_samples": 120000,
                "output_dir": "./results/pythia-70M-full",
                "num_train_epochs": 3,
                "save_steps": 3750
            }
            output_dir = train_70m(**train_70m_params)
            print(f"70M model trained and saved to: {output_dir}")

    if args.get_trajectories:
        trajectories_params = {
            "checkpoint_path": "./results/pythia-70M-full",
            "dataset_name": "TIGER-Lab/MathInstruct",
            "n_samples": 120000,
            "run_name": None
        }
        print(f"\nGenerating loss trajectories from {trajectories_params['checkpoint_path']}")
        get_trajectories(**trajectories_params)

        """
        NEW S2L + CURRICULUM ALGORITHMS (RECOMMENDED):
        - "avg_loss_curriculum": S2L clustering + curriculum by average loss
          Ranks samples by mean loss across checkpoints. Lower loss = easier.
          Returns n_samples per difficulty level (easy, medium, hard).

        - "loss_decrease_curriculum": S2L clustering + curriculum by loss improvement
          Ranks samples by (final_loss - initial_loss). Less improvement = easier.
          Returns n_samples per difficulty level (easy, medium, hard).

        ORIGINAL S2L (NO CURRICULUM):
        - "s2l_select": Original S2L algorithm - cluster by loss trajectories,
          sample evenly from all clusters. No curriculum learning.
        """
    if args.config_selection is not None:
        data_selection(config_file=args.config_selection)
    else:
        selection_params = {
            "losses_dir": "./results/pythia-70M-full/losses",
            "dataset_name": "TIGER-Lab/MathInstruct",
            "n_samples": 36_000,  # 30% of the data
            "algorithm": "s2l_select",
            "dataset_cutoff": 120000,
            "num_epochs": 3,
            "output_file": "./results/data/70M-S2L.json"
        }
        data_selection(**selection_params)


    if args.train_410m:
        if args.config_410m is not None:
            train_410m(config_file=args.config_410m)
        else:
            train_410m_params = {
                "model_path": "EleutherAI/pythia-410m-deduped",
                "dataset_name": "./results/data/70M-S2L.json",  # this needs to be the same path as the output file from before
                "n_samples": -1,  # Use all selected samples
                "output_dir": "./results/pythia-410M-s2l",
                "num_train_epochs": 1,
                "save_steps": 3750
            }
            train_410m(**train_410m_params)
