import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from gnn_model import SquatGNN, create_graph_from_sequence
from torch_geometric.data import Data, Batch


def prepare_data(sequence, label):
    """Create a PyTorch Geometric Data object with label"""
    graph = create_graph_from_sequence(sequence)
    graph.y = torch.tensor([label], dtype=torch.long)
    return graph


def train_model(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        # Move batch to device
        batch = batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model.forward_data(batch)

        # Calculate loss
        loss = F.nll_loss(output, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        total_loss += loss.item()
        pred = output.max(1)[1]
        correct += pred.eq(batch.y).sum().item()
        total += batch.y.size(0)

    accuracy = 100. * correct / total
    return total_loss / len(train_loader), accuracy


if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    valid_sequences = np.load('preprocessed_valid.npy')
    invalid_sequences = np.load('preprocessed_invalid.npy')

    print("Creating graphs...")
    # Create graphs with labels
    valid_graphs = [prepare_data(seq, 1) for seq in valid_sequences]
    invalid_graphs = [prepare_data(seq, 0) for seq in invalid_sequences]

    # Combine all graphs
    all_graphs = valid_graphs + invalid_graphs

    # Create data loader
    train_loader = DataLoader(all_graphs, batch_size=32, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SquatGNN(input_dim=132, hidden_dim=64, output_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train
    print("Starting training...")
    num_epochs = 1000
    best_accuracy = 0

    try:
        for epoch in range(num_epochs):
            loss, accuracy = train_model(train_loader, model, optimizer, device)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1:03d}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')

                # Save best model
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), 'best_squat_classifier.pth')

        print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Print more debug information
        print("\nDebug information:")
        print(f"Number of valid sequences: {len(valid_sequences)}")
        print(f"Number of invalid sequences: {len(invalid_sequences)}")
        print(f"Shape of first valid sequence: {valid_sequences[0].shape}")
        print(f"Shape of first invalid sequence: {invalid_sequences[0].shape}")