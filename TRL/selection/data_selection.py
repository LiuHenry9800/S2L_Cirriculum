import torch
import numpy as np
import faiss
import glob
import os
import json
import yaml
import argparse
from datasets import load_dataset

class SelectionConfig:
    def __init__(self, config_file):
        self.losses_dir = None
        self.dataset_name = "TIGER-Lab/MathInstruct"
        self.dataset_cutoff = 120000
        self.n_samples = 5000
        self.num_epochs = 4
        self.algorithm = "s2l_select"
        self.output_file = "selected_data.json"
        
        if config_file:
            with open(config_file) as f:
                config_dict = yaml.safe_load(f)
                for key, value in config_dict.items():
                    setattr(self, key, value)

#main loop
def select_training_data(config:SelectionConfig):
    split = f"train[:{config.dataset_cutoff}]" if config.dataset_cutoff > -1 else "train"
    dataset = load_dataset(config.dataset_name, split=split)

    sources = np.array([data['source'] for data in dataset])
    
    if os.path.exists(config.losses_dir):
        print(f"All files in directory: {os.listdir(config.losses_dir)}")

    losses = []
    loss_files = glob.glob(os.path.join(config.losses_dir, "ckpt*_loss.pt"))
    print(loss_files)
    loss_files = sorted(loss_files, key=lambda x: int(x.split('ckpt')[1].split('_')[0]))
    for loss_file in loss_files:
        print(f"found loss file: {loss_file}")
        losses.append(torch.stack(torch.load(loss_file)))
    
    losses = torch.stack(losses).t()
    losses[torch.isnan(losses)] = 0
    print(f"Loss traj shape: {losses.shape}")
    
    #decide the algo
    if config.algorithm == 's2l_select':
        # Original S2L - no curriculum
        selected_indices = select_with_algorithm(losses,
                                                sources,
                                                config.n_samples,
                                                n_clusters=100,
                                                num_epochs=None,
                                                algorithm=s2l_algo)
    elif config.algorithm == 'avg_loss_curriculum':
        # S2L + curriculum with average loss heuristic
        selected_indices = select_with_algorithm(losses,
                                                sources,
                                                config.n_samples,
                                                n_clusters=100,
                                                num_epochs=config.num_epochs,
                                                algorithm=avg_loss_curriculum)
    elif config.algorithm == 'loss_decrease_curriculum':
        # S2L + curriculum with loss decrease heuristic
        selected_indices = select_with_algorithm(losses,
                                                sources,
                                                config.n_samples,
                                                n_clusters=100,
                                                num_epochs=config.num_epochs,
                                                algorithm=loss_decrease_curriculum)
    elif config.algorithm == 'avg_loss_select':
        # Old curriculum helper (kept for backwards compatibility)
        selected_indices = select_with_algorithm(losses,
                                                sources,
                                                config.n_samples,
                                                n_clusters=100,
                                                num_epochs=config.num_epochs,
                                                algorithm=avg_loss_algo)
    elif config.algorithm == "overall_loss_decrease_select":
        # Old curriculum helper (kept for backwards compatibility)
        selected_indices = select_with_algorithm(losses,
                                                sources,
                                                config.n_samples,
                                                n_clusters=100,
                                                num_epochs=config.num_epochs,
                                                algorithm=overall_loss_decrease_algo)
    else:
        raise ValueError(f"Unknown selection algorithm: {config.algorithm}")
    
    #idc to data
    selected_data = [dataset[int(idx)] for idx in selected_indices]
    
    with open(config.output_file, 'w') as f:
        json.dump(selected_data, f, indent=2)
    
    print(f"Total selected: {len(selected_data)} samples, saved to {config.output_file}")
    return selected_indices

#main loop chooses an algorithm and selects all the data
def select_with_algorithm(losses, sources, n_samples, n_clusters, num_epochs, algorithm):
    unique_sources, source_counts = np.unique(sources, return_counts=True)
    sorted_source_idx = np.argsort(source_counts)
    
    selected_indices = []
    remaining = n_samples
    
    for i in range(len(sorted_source_idx)):
        source = unique_sources[sorted_source_idx[i]]
        source_indices = np.where(sources == source)[0]
        n_per_source = remaining // (len(sorted_source_idx) - i)
        
        if len(source_indices) > n_per_source:
            source_losses = losses[source_indices]
            selected = algorithm(source_losses, n_per_source, n_clusters, num_epochs)
            selected_indices.extend(source_indices[selected].tolist())
            remaining -= n_per_source
        else:
            selected_indices.extend(source_indices.tolist())
            remaining -= len(source_indices)
    
    return selected_indices

# use for any cirriculum selection
def curriculum_select_helper(losses, n_samples, n_clusters, num_epochs, ranking_fn):
    """Generic curriculum selection with pluggable ranking function"""
    kmeans = faiss.Kmeans(losses.shape[1], n_clusters, niter=20, verbose=False)
    kmeans.train(losses.numpy())
    _, cluster_labels = kmeans.index.search(losses.numpy(), 1)
    
    clusters, counts = np.unique(cluster_labels, return_counts=True)
    
    cluster_scores = []
    for cluster_id in clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        score = ranking_fn(losses[cluster_indices])
        cluster_scores.append((cluster_id, score))
    
    cluster_scores.sort(key=lambda x: x[1])
    sorted_clusters = np.array([c[0] for c in cluster_scores])
    
    samples_per_epoch = n_samples // num_epochs
    selected = []
    
    for epoch in range(num_epochs):

        #btw, make sure cluster count divisible by num epochs
        pool_size = int(len(sorted_clusters) * (epoch + 1) / num_epochs)
        available_clusters = sorted_clusters[:pool_size]
        large_clusters = available_clusters[np.isin(available_clusters, clusters[counts > 2])]
        
        remaining = samples_per_epoch
        
        for i in range(len(large_clusters)):
            cluster_id = large_clusters[i]
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            n_per_cluster = remaining // (len(large_clusters) - i)
            
            if len(cluster_indices) > n_per_cluster:
                idcs = np.random.choice(cluster_indices, n_per_cluster, replace=False)
            else:
                idcs = cluster_indices
            
            selected.extend(idcs)
            remaining -= len(idcs)
        
        if remaining > 0:
            small_clusters = available_clusters[np.isin(available_clusters, clusters[counts <= 2])]
            small_indices = np.where(np.isin(cluster_labels, small_clusters))[0]
            if len(small_indices) > 0:
                sel = np.random.choice(small_indices, min(remaining, len(small_indices)), replace=False)
                selected.extend(sel)
    
    return np.array(selected[:n_samples])

# avg loss selection
def avg_loss_algo(losses, n_samples, n_clusters, num_epochs):
    def avg_loss(loss_amts):
        return loss_amts.mean()
    return curriculum_select_helper(
        losses, n_samples, n_clusters, num_epochs,
        ranking_fn=avg_loss
    )
#(last - first loss) take the mean of all samples in that cluster
def overall_loss_decrease_algo(losses, n_samples, n_clusters, num_epochs):
    def overall_decrease(loss_amts):
        return (loss_amts[:, -1] - loss_amts[:, 0]).mean()
    return curriculum_select_helper(
        losses, n_samples, n_clusters, num_epochs,
        ranking_fn=overall_decrease
    )


# S2L curriculum learning with difficulty heuristics
def s2l_curriculum_algo(losses, n_samples, n_clusters, num_epochs, ranking_fn):
    """
    S2L clustering + curriculum learning within each cluster

    Process:
    1. Do S2L clustering (cluster by loss trajectories)
    2. Sample evenly from ALL clusters
    3. Within each cluster, rank samples by difficulty using ranking_fn
    4. Pick n_samples easy, then n_samples medium, then n_samples hard
    5. Concatenate: [easy, medium, hard]

    Args:
        losses: Loss trajectories for samples
        n_samples: Samples to select PER difficulty level (total = n_samples * num_epochs)
        n_clusters: Number of clusters for K-means
        num_epochs: Number of difficulty levels (e.g., 3 for easy/medium/hard)
        ranking_fn: Function to rank sample difficulty (takes single sample losses, returns score)

    Returns:
        Array of selected indices, ordered by difficulty (easy first, hard last)
    """
    # Step 1: S2L clustering (always the same)
    print(f"Performing S2L clustering with {n_clusters} clusters...")
    kmeans = faiss.Kmeans(losses.shape[1], n_clusters, niter=20, verbose=False)
    kmeans.train(losses.numpy())
    _, cluster_labels = kmeans.index.search(losses.numpy(), 1)

    clusters, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Created {len(clusters)} clusters")

    # Filter for larger clusters (S2L strategy)
    large_clusters = clusters[counts > 2]
    small_clusters = clusters[counts <= 2]

    all_selected = []

    # Step 2: For each difficulty level (easy, medium, hard)
    # Sample evenly from ALL clusters, but pick easy/medium/hard samples within each cluster
    for epoch in range(num_epochs):
        difficulty_name = ["Easy", "Medium", "Hard"][epoch] if num_epochs == 3 else f"Level {epoch + 1}"
        print(f"\nSelecting {n_samples} {difficulty_name} samples...")

        epoch_selected = []
        remaining = n_samples

        # Sample evenly from large clusters
        samples_per_cluster = remaining // len(large_clusters) if len(large_clusters) > 0 else 0

        for cluster_id in large_clusters:
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_losses = losses[cluster_indices]

            # Rank samples within this cluster by difficulty
            sample_scores = []
            for i, idx in enumerate(cluster_indices):
                score = ranking_fn(cluster_losses[i])
                sample_scores.append((idx, score))

            # Sort by difficulty (ascending = easy to hard)
            sample_scores.sort(key=lambda x: x[1])
            sorted_indices = np.array([s[0] for s in sample_scores])

            # Determine which samples to pick from this cluster for this difficulty level
            total_cluster_samples = len(sorted_indices)
            start_idx = int(total_cluster_samples * epoch / num_epochs)
            end_idx = int(total_cluster_samples * (epoch + 1) / num_epochs)
            difficulty_samples = sorted_indices[start_idx:end_idx]

            # Pick samples_per_cluster from this difficulty level
            n_to_pick = min(samples_per_cluster, len(difficulty_samples))
            if n_to_pick > 0:
                selected = np.random.choice(difficulty_samples, n_to_pick, replace=False)
                epoch_selected.extend(selected)
                remaining -= n_to_pick

        # Fill remaining from small clusters (randomly)
        if remaining > 0 and len(small_clusters) > 0:
            small_indices = np.where(np.isin(cluster_labels, small_clusters))[0]
            if len(small_indices) > 0:
                n_to_pick = min(remaining, len(small_indices))
                sel = np.random.choice(small_indices, n_to_pick, replace=False)
                epoch_selected.extend(sel)

        all_selected.extend(epoch_selected)
        print(f"  Selected {len(epoch_selected)} samples ({difficulty_name})")

    print(f"\nTotal selected: {len(all_selected)} samples")
    return np.array(all_selected)


# Curriculum with average loss heuristic
def avg_loss_curriculum(losses, n_samples, n_clusters, num_epochs):
    def avg_loss(sample_losses):
        # Single sample: shape [num_checkpoints]
        return sample_losses.mean()
    return s2l_curriculum_algo(losses, n_samples, n_clusters, num_epochs, ranking_fn=avg_loss)


# Curriculum with loss decrease heuristic
def loss_decrease_curriculum(losses, n_samples, n_clusters, num_epochs):
    def loss_decrease(sample_losses):
        # Single sample: shape [num_checkpoints]
        # Negated so smaller = easier (less improvement = already easy)
        return -(sample_losses[-1] - sample_losses[0])
    return s2l_curriculum_algo(losses, n_samples, n_clusters, num_epochs, ranking_fn=loss_decrease)


# base s2l
def s2l_algo(losses, n_samples, n_clusters,num_epochs):
    kmeans = faiss.Kmeans(losses.shape[1], n_clusters, niter=20, verbose=False)
    kmeans.train(losses.numpy())
    
    _, cluster_labels = kmeans.index.search(losses.numpy(), 1)
    
    clusters, counts = np.unique(cluster_labels, return_counts=True)
    sorted_idx = np.argsort(counts)
    
    sorted_idx = sorted_idx[counts[sorted_idx] > 2]
    
    selected = []
    remaining = n_samples
    
    print("Take from clusters with size > 2")
    # Add to selected from from smallest to largest
    for i in range(len(sorted_idx)):
        cluster_id = clusters[sorted_idx[i]]
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        n_per_cluster = remaining // (len(sorted_idx) - i)
        
        if len(cluster_indices) > n_per_cluster:
            idcs = np.random.choice(cluster_indices, n_per_cluster, replace=False)
        else:
            idcs = cluster_indices
        
        selected.extend(idcs)
        remaining -= len(idcs)
    
    print("Take from clusters with size <= 2")
    if remaining > 0:
        small_clusters = clusters[counts <= 2]
        small_indices = np.where(np.isin(cluster_labels, small_clusters))[0]
        if len(small_indices) > 0:
            sel = np.random.choice(small_indices, min(remaining, len(small_indices)), replace=False)
            selected.extend(sel)
    
    return np.array(selected[:n_samples])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = SelectionConfig(args.config)
    select_training_data(config)
