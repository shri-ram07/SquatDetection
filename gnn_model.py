import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


def create_graph_from_sequence(sequence):
    """
    Convert a sequence of keypoints to a graph structure.

    Args:
        sequence: numpy array of shape (frames, features)
    Returns:
        Data: PyTorch Geometric Data object
    """
    # Convert sequence to tensor
    nodes = torch.tensor(sequence, dtype=torch.float)

    # Create edges between consecutive frames
    num_frames = sequence.shape[0]
    edges = []

    # Connect each frame with next frame
    for i in range(num_frames - 1):
        edges.extend([[i, i + 1], [i + 1, i]])  # bidirectional edges
        edges.append([i, i])  # self-loop

    # Add self-loop for last frame
    edges.append([num_frames - 1, num_frames - 1])

    # Convert edges to tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    return Data(x=nodes, edge_index=edge_index)


class SquatGNN(torch.nn.Module):
    """
    Graph Neural Network for squat classification.

    Args:
        input_dim (int): Number of input features per node
        hidden_dim (int): Number of hidden features
        output_dim (int): Number of output classes (2 for binary classification)
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SquatGNN, self).__init__()

        # First Graph Convolution Layer
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # Second Graph Convolution Layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Final classification layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

        # Dropout rate
        self.dropout_rate = 0.5

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for ONNX compatibility.

        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignments for nodes

        Returns:
            torch.Tensor: Classification logits
        """
        # First conv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Second conv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Handle batch information
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Classification
        x = self.fc(x)

        # Return log probabilities
        return F.log_softmax(x, dim=1)

    def forward_data(self, data):
        """
        Forward pass for PyTorch Geometric Data objects.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            torch.Tensor: Classification logits
        """
        return self.forward(data.x, data.edge_index, data.batch)


def create_model(input_dim=132, hidden_dim=64, output_dim=2):
    """
    Helper function to create a model instance with default parameters.

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension (number of classes)

    Returns:
        SquatGNN: Initialized model
    """
    model = SquatGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    return model


# Example usage and testing code
if __name__ == "__main__":
    # Create dummy data for testing
    num_frames = 100
    num_features = 132

    # Create dummy sequence
    dummy_sequence = torch.randn(num_frames, num_features)

    # Create graph from sequence
    graph = create_graph_from_sequence(dummy_sequence.numpy())

    # Create model
    model = create_model()

    # Test forward pass
    model.eval()
    with torch.no_grad():
        # Test with graph
        output_graph = model.forward_data(graph)
        print("Output shape from graph:", output_graph.shape)

        # Test with tensors
        output_tensor = model(
            graph.x,
            graph.edge_index,
            torch.zeros(graph.x.size(0), dtype=torch.long)
        )
        print("Output shape from tensors:", output_tensor.shape)

    print("\nModel architecture:")
    print(model)