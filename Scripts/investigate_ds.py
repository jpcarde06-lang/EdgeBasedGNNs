import os
from tqdm import tqdm
import torch
from torch_geometric.datasets import QM9, MalNetTiny
from Misc.config import config
from Misc.QM_dataset import QM_Dataset

def analyze_dataset(dataset, name="Dataset"):
    
    if "x" not in dataset[0]:
        nr_features_x = 0
    else:
        nr_features_x = dataset[0].x.shape[1]
        
    if "edge_attr" not in dataset[0]:
        nr_features_e = 0
    else:
        nr_features_e = dataset[0].edge_attr.shape[1]
    

    max_values_x = [0 for _ in range(nr_features_x)]
    max_values_e = [0 for _ in range(nr_features_e)]
    total_nodes = 0

    for graph in tqdm(dataset, desc=f"Processing {name}"):
        total_nodes += graph.num_nodes

        for i in range(nr_features_x):
            max_value = torch.max(graph.x[:, i])
            if max_value > max_values_x[i]:
                max_values_x[i] = int(max_value.item())
        for i in range(nr_features_e):
            max_value = torch.max(graph.edge_attr[:, i])
            if max_value > max_values_e[i]:
                max_values_e[i] = int(max_value.item())

    avg_nodes = total_nodes / len(dataset)
    print(f"{name} - Vertex features: {max_values_x}")
    print(f"{name} - Edge features: {max_values_e}")
    print(f"{name} - Average number of nodes: {avg_nodes:.2f}")

def main():
    # QM9
    # qm9_dir = os.path.join(config.DATA_PATH, "QM9", "Original")
    # qm9_dataset = QM9(root=qm9_dir)
    # analyze_dataset(qm9_dataset, "QM9")

    # QMD 
    # qmd_task = "0"  # replace with your specific QMD task
    # qmd_dir = os.path.join(config.DATA_PATH, f"QMD/{qmd_task}")
    # qmd_dataset = QM_Dataset(root=qmd_dir)
    # analyze_dataset(qmd_dataset, f"QMD-{qmd_task}")
    
    # MalNet Tiny (all splits)
    malnet_dir = os.path.join(config.DATA_PATH, "MalNetTiny")
    for split in ["train", "val", "test"]:
        malnet_dataset = MalNetTiny(root=malnet_dir, split=split)
        analyze_dataset(malnet_dataset, f"MalNetTiny-{split}")

if __name__ == "__main__":
    main()
