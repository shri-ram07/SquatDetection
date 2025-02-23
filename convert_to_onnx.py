import torch
import numpy as np
from gnn_model import SquatGNN


class ONNXSquatClassifier(torch.nn.Module):
    def __init__(self, gnn_model):
        super(ONNXSquatClassifier, self).__init__()
        self.gnn_model = gnn_model

    def forward(self, x):
        # Create edge_index for sequence data
        # Assuming x has shape [batch_size, sequence_length, features]
        batch_size, seq_len, _ = x.shape

        # Create edges between consecutive frames
        edge_index = []
        for i in range(seq_len - 1):
            edge_index.extend([[i, i + 1], [i + 1, i]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()

        # Reshape input for GNN
        x = x.reshape(-1, x.size(-1))  # [batch_size * sequence_length, features]

        # Create batch index
        batch = torch.zeros(x.size(0), dtype=torch.long)

        # Forward pass through GNN
        return self.gnn_model(x, edge_index, batch)


def convert_to_onnx(model_path, output_path, input_shape):
    # Load the GNN model
    gnn_model = SquatGNN(input_dim=132, hidden_dim=64, output_dim=2)
    gnn_model.load_state_dict(torch.load(model_path))
    gnn_model.eval()

    # Create ONNX wrapper
    onnx_model = ONNXSquatClassifier(gnn_model)
    onnx_model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model converted to ONNX and saved to {output_path}")


if __name__ == "__main__":
    # Configuration
    model_path = 'best_squat_classifier.pth'
    output_path = 'squat_classifier.onnx'
    input_shape = (1, 100, 132)  # [batch_size, sequence_length, features]

    # Convert model
    try:
        convert_to_onnx(model_path, output_path, input_shape)
        print("Conversion successful!")

        # Verify the model
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")