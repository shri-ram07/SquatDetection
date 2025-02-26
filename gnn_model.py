# improved_gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm


class ImprovedGNNClassifier(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, num_classes=2, dropout=0.5):
        super(ImprovedGNNClassifier, self).__init__()
        # First GCN layer with BatchNorm and Dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        # Second GCN layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        # Third GCN layer for extra depth
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        # Final linear layer for classification
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Global pooling to obtain graph-level representation
        x = global_mean_pool(x, batch)
        # Final classification layer
        x = self.lin(x)
        return x
