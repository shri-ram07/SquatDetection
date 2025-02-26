===============================================================================
                         SQUAT RECOGNITION WITH GNN
===============================================================================

Overview
--------
This project implements a squat classification system that uses human 
pose estimation and Graph Neural Networks (GNN) to detect whether a squat 
is "Correct" or "Incorrect". The pipeline leverages the MoveNet model for 
keypoint extraction, constructs spatio-temporal graphs from video data, 
and trains advanced GNN architectures (including transformer-based models) 
to learn discriminative features for squat classification.

Key Features:
- **Pose Estimation:** Uses MoveNet (SinglePose Thunder) from TensorFlow Hub 
  to extract 17 keypoints per frame.
- **Graph Construction:** Builds spatio-temporal graphs where nodes represent 
  keypoints (x, y, confidence) and edges capture both spatial (body-skeleton) 
  and temporal (across frames) relationships.
- **Model Training:** Implements improved GNN models (ImprovedGNNClassifier and 
  AdvancedGNNClassifier) using PyTorch Geometric.
- **Real-Time Inference:** Supports live squat detection via a webcam feed.
- **Model Deployment:** Provides scripts to export trained models to ONNX format.

Directory Structure
-------------------
The repository is organized as follows:


    ├── main__code.ipynb        #main code with all things, 
    │                          # and extracting keypoints.
    ├── graph_utils.py          # Functions to build spatio-temporal graphs from keypoints.
    ├── squat_dataset.py        # Custom PyTorch Geometric Dataset to process videos 
    │                          # and save/load "squat_dataset.pt".
    ├── gnn_model.py   # Definition of the ImprovedGNNClassifier.
    ├── test.py  # Script to test the improved GNN model on preprocessed data.
    │                          # Script for real-time squat detection (live feed).
    ├── export_onnx.py          # Script to export the trained model to ONNX format.
    ├── README.txt              # This file.
    └── requirements.txt        # (Optional) List of dependencies.

Installation
------------
1. **Clone the Repository:**
   git clone <repository_url>
   cd <repository_directory>

2. **Install Dependencies:**
   Ensure you have Python 3.7+ installed. Then install required packages:
   
       pip install tensorflow tensorflow-hub opencv-python numpy torch
       pip install torch-geometric
       
   Note: For PyTorch Geometric, refer to the official installation instructions 
   (https://pytorch-geometric.readthedocs.io) as additional dependencies may be needed.

3. **(Optional) Google Colab:**
   - Mount your Google Drive if your data is stored there.
   - Use `%load_ext autoreload` and `%autoreload 2` in your notebook for automatic
     module reloading.

Data Preparation
----------------
Place your squat videos in the following folder structure:

    video_dataset/
       ├── Correct/
       │      video1.mp4, video2.mp4, ...
       └── Incorrect/
              videoA.mp4, videoB.mp4, ...

The SquatDataset class (in squat_dataset.py) will process these videos using 
MoveNet (via movenet_utils.py) to extract keypoints and build spatio-temporal 
graphs. The processed data is saved as "squat_dataset.pt" for fast reloading.

Training the Model
------------------
1. **Run Preprocessing and Training:**
   Use the training script to load the preprocessed data and train the model.
   
       python train.py
       
   The script splits the dataset (e.g., 80/20 for training/testing), trains the 
   model, and saves the weights (e.g., "improved_gnn_model_from_pt.pth").

2. **Hyperparameter Tuning:**
   Experiment with learning rates, dropout rates, hidden dimensions, and 
   augmentation (e.g., on-the-fly noise added to keypoints) in the training script.

Testing the Model
-----------------
1. **Batch Testing:**
   To evaluate the trained model on the test set from "squat_dataset.pt", run:
   
       python test_improved_model.py
   
   This script loads the saved model weights and prints the test accuracy.

2. **Live (Real-Time) Inference:**
   For live squat detection via your webcam, run:
   
       python test_live_improved_gnn.py
   
   The script captures video from your webcam, uses MoveNet to extract keypoints, 
   builds a sliding window spatio-temporal graph, and displays a real-time prediction 
   ("Correct" vs. "Incorrect") overlaid on the video feed. Press "q" to exit.

Model Deployment: ONNX Conversion
----------------------------------
To convert the trained PyTorch model to ONNX format for deployment, run:
   
       python export_onnx.py

This script loads your model from "improved_gnn_model_from_pt.pth" (or another weight file),
creates dummy inputs for dynamic graph shapes, and exports the model as "improved_gnn_model.onnx".

Advanced Model Architectures
-----------------------------
- **ImprovedGNNClassifier:** A GNN model with enhanced regularization, batch normalization,
  and dropout.
- **AdvancedGNNClassifier:** A transformer-based model using TransformerConv layers, multi-head 
  attention, and batch normalization to better capture complex spatial relationships.
- **Hybrid Spatio-Temporal Model:** (Optional) A model that combines spatial GCN layers with 
  temporal modules (e.g., LSTM) to capture the dynamics of squat movements.

User Guide & Troubleshooting
----------------------------
- **Running Scripts:**  
  Run the provided Python scripts from the command line or in your preferred IDE.
  
- **Data Issues:**  
  Ensure that the "video_dataset" folder is correctly structured. If the preprocessed 
  file ("squat_dataset.pt") is outdated, delete it to force reprocessing.
  
- **Performance Tuning:**  
  Monitor training logs and adjust hyperparameters. Consider using data augmentation 
  or ensembling methods if accuracy is lower than expected.
  
- **Live Inference:**  
  Make sure your webcam is working. Adjust the `WINDOW_SIZE` parameter in the live testing 
  script if the predictions seem unstable.
  
- **ONNX Deployment:**  
  Verify dummy input shapes in export_onnx.py match typical graph sizes from your dataset.



===============================================================================
