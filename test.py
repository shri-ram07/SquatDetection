import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
from torch_geometric.data import Data
from gnn_model import ImprovedGNNClassifier  # Your improved model class
from graph_utils import build_graph_from_keypoints  # Function to build a graph from a list of keypoints

# --- MoveNet Utilities ---
def load_movenet_model():
    model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    model = hub.load(model_url)
    return model

def preprocess_frame(frame, target_size=(256, 256)):
    # Resize and convert BGR to RGB.
    img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Cast to int32 (required by MoveNet); do not normalize.
    img = img.astype(np.int32)
    input_img = np.expand_dims(img, axis=0)
    return input_img

def extract_keypoints_from_frame(movenet_model, frame):
    input_img = preprocess_frame(frame)
    input_tensor = tf.constant(input_img, dtype=tf.int32)
    outputs = movenet_model.signatures['serving_default'](input=input_tensor)
    keypoints = outputs['output_0'].numpy().squeeze()  # Expected shape: (17, 3)
    return keypoints

# --- Real-Time Inference Settings ---
WINDOW_SIZE = 10  # Number of consecutive frames to form a spatio-temporal graph
keypoints_buffer = []  # Buffer to hold keypoints from consecutive frames

# --- Load Models ---
print("Loading MoveNet model...")
movenet_model = load_movenet_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize and load your improved GNN model
model = ImprovedGNNClassifier(in_channels=3, hidden_channels=128, num_classes=2, dropout=0.5).to(device)
model.load_state_dict(torch.load("improved_gnn_model_from_pt.pth", map_location=device))
model.eval()

# --- Start Live Video Capture ---
cap = cv2.VideoCapture(r"video_dataset/Correct/0918_squat_000001.mp4")  # Use 0 for webcam; replace with a file path for a video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract keypoints from the current frame using MoveNet.
    keypoints = extract_keypoints_from_frame(movenet_model, frame)
    keypoints_buffer.append(keypoints)

    # Once the buffer is full, build a spatio-temporal graph and run inference.
    if len(keypoints_buffer) == WINDOW_SIZE:
        graph_data = build_graph_from_keypoints(keypoints_buffer)
        # Create a batch vector where all nodes belong to one graph.
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
        graph_data = graph_data.to(device)

        with torch.no_grad():
            output = model(graph_data.x, graph_data.edge_index, graph_data.batch)
            pred = torch.argmax(output, dim=1).item()  # 0: Incorrect, 1: Correct

        # Overlay prediction on the current frame.
        pred_text = "Correct" if pred == 1 else "Incorrect"
        color = (0, 255, 0) if pred == 1 else (0, 0, 255)
        cv2.putText(frame, f"Squat: {pred_text}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2, cv2.LINE_AA)

        # Remove the oldest set of keypoints to maintain the sliding window.
        keypoints_buffer.pop(0)

    # Display the video frame with overlay.
    cv2.imshow("Live Squat Detection (Improved GNN)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
