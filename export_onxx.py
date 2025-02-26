import torch
from gnn_model import ImprovedGNNClassifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedGNNClassifier(in_channels=3, hidden_channels=128, num_classes=2, dropout=0.5).to(device)
model.load_state_dict(torch.load("improved_gnn_model_from_pt.pth", map_location=device))
model.eval()
# Create dummy inputs:
# - dummy_x: Node features with shape (num_nodes, 3)
# - dummy_edge_index: Graph connectivity with shape (2, num_edges)
# - dummy_batch: Batch vector with shape (num_nodes,) (all nodes in one graph)
dummy_x = torch.randn(50, 3, device=device)           # For example, 50 nodes with 3 features each.
dummy_edge_index = torch.randint(0, 50, (2, 200), device=device)  # Example connectivity (200 edges).
dummy_batch = torch.zeros(50, dtype=torch.long, device=device)     # All nodes belong to a single graph.

# Export the model to ONNX format.
torch.onnx.export(
    model,                                           # model being run
    (dummy_x, dummy_edge_index, dummy_batch),        # model inputs (a tuple)
    "improved_gnn_model.onnx",                       # where to save the model
    input_names=["x", "edge_index", "batch"],        # input names
    output_names=["output"],                         # output names
    dynamic_axes={
        "x": {0: "num_nodes"},                       # variable number of nodes
        "edge_index": {1: "num_edges"},              # variable number of edges
        "batch": {0: "num_nodes"}                    # batch vector length can vary
    },
    opset_version=11                                  # Specify the ONNX opset version
)

print("Model has been exported to improved_gnn_model.onnx")