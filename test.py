import torch
from gnn_model import SquatGNN, create_graph_from_sequence
import numpy as np


def test_model(model, test_sequence, device):
    model.eval()
    with torch.no_grad():
        # Create graph from sequence
        graph = create_graph_from_sequence(test_sequence)
        graph = graph.to(device)

        # Get prediction
        output = model(graph)
        pred = output.max(1)[1]
        return pred.item()


def test_multiple_sequences(model, test_sequences, device):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for sequence in test_sequences:
            # Create graph from sequence
            graph = create_graph_from_sequence(sequence)
            graph = graph.to(device)

            # Get prediction
            output = model(graph)
            pred = output.max(1)[1]

            # For valid sequences, label should be 1
            correct += (pred.item() == 1)
            total += 1

    return (correct / total) * 100


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = SquatGNN(input_dim=132, hidden_dim=64, output_dim=2)
    model.load_state_dict(torch.load('best_squat_classifier.pth'))
    model.to(device)
    model.eval()

    # Load test data (using some sequences from our preprocessed data)
    print("Loading test data...")
    valid_sequences = np.load('preprocessed_valid.npy')
    invalid_sequences = np.load('preprocessed_invalid.npy')

    # Take last 10 sequences from each for testing
    test_valid = valid_sequences[-10:]
    test_invalid = invalid_sequences[-10:]

    # Test on valid sequences
    print("\nTesting on valid sequences...")
    valid_accuracy = test_multiple_sequences(model, test_valid, device)
    print(f"Accuracy on valid sequences: {valid_accuracy:.2f}%")

    # Test on invalid sequences
    print("\nTesting on invalid sequences...")
    invalid_accuracy = test_multiple_sequences(model, test_invalid, device)
    print(f"Accuracy on invalid sequences: {100 - invalid_accuracy:.2f}%")  # Should predict 0

    # Overall accuracy
    overall_accuracy = (valid_accuracy + (100 - invalid_accuracy)) / 2
    print(f"\nOverall accuracy: {overall_accuracy:.2f}%")

    # Test a single sequence
    print("\nTesting single sequence...")
    test_sequence = valid_sequences[0]  # Take first valid sequence
    result = test_model(model, test_sequence, device)
    print("Prediction for single sequence:", "Valid" if result == 1 else "Invalid")