# graph_utils.py
import torch
from torch_geometric.data import Data
import numpy as np
# Define the skeleton connections (indices based on COCO order)
COCO_SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11),
    (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

def build_graph_from_keypoints(keypoints_sequence):
    """
    keypoints_sequence: list of numpy arrays of shape (17,3) for each frame.
    Returns a torch_geometric.data.Data object representing the spatio-temporal graph.
    """
    num_frames = len(keypoints_sequence)
    num_keypoints = keypoints_sequence[0].shape[0]  # usually 17
    # Create node features list.
    node_features = []
    for frame in keypoints_sequence:
        # Each frame has shape (17, 3)
        node_features.append(frame)
    # Stack into (num_frames * 17, 3)
    x = torch.tensor(np.vstack(node_features), dtype=torch.float)

    edge_index_list = []

    # Build spatial edges for each frame.
    for t in range(num_frames):
        base_idx = t * num_keypoints
        for (i, j) in COCO_SKELETON:
            src = base_idx + i
            dst = base_idx + j
            # Add bidirectional edges.
            edge_index_list.append([src, dst])
            edge_index_list.append([dst, src])

    # Build temporal edges: connect the same joint between consecutive frames.
    for t in range(num_frames - 1):
        for i in range(num_keypoints):
            src = t * num_keypoints + i
            dst = (t + 1) * num_keypoints + i
            edge_index_list.append([src, dst])
            edge_index_list.append([dst, src])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # Create and return a Data object. (Assume label is provided later.)
    data = Data(x=x, edge_index=edge_index)
    return data