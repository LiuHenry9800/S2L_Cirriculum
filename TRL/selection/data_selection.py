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
        selected_indices = select_with_algorithm(losses,
                                                sources,
                                                config.n_samples, 
                                                n_clusters=100, 
                                                num_epochs=None, 
                                                algorithm=s2l_cluster_select)
    elif config.algorithm == 'avg_loss_select':
        selected_indices = select_with_algorithm(losses,
                                                sources,
                                                config.n_samples, 
                                                n_clusters=100, 
                                                num_epochs=config.num_epochs, 
                                                algorithm=avg_cluster_select)
    else:
        raise ValueError(f"Unknown selection algorithm: {config.algorithm}")
    
    #idc to data
    selected_data = [dataset[int(idx)] for idx in selected_indices]
    
    with open(config.output_file, 'w') as f:
        json.dump(selected_data, f, indent=2)
    
    print(f"Total selected: {len(selected_data)} samples, saved to {config.output_file}")
    return selected_indices

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

def avg_cluster_select(losses, n_samples, n_clusters, num_epochs):
    kmeans = faiss.Kmeans(losses.shape[1], n_clusters, niter=20, verbose=False)
    kmeans.train(losses.numpy())
    _, cluster_labels = kmeans.index.search(losses.numpy(), 1)
    
    avg_losses = losses.mean(dim=1).numpy()
    clusters, counts = np.unique(cluster_labels, return_counts=True)
    
    cluster_avg_losses = []
    for cluster_id in clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_avg_loss = avg_losses[cluster_indices].mean()
        cluster_avg_losses.append((cluster_id, cluster_avg_loss))
    
    cluster_avg_losses.sort(key=lambda x: x[1])
    sorted_clusters = np.array([c[0] for c in cluster_avg_losses])
    
    samples_per_epoch = n_samples // num_epochs
    selected = []
    
    for epoch in range(num_epochs):
        #btw, make sure num clusters(usually 100) divisble by num_epcohs
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


def s2l_cluster_select(losses, n_samples, n_clusters,num_epochs):
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
