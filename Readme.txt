SQUAT ANALYSIS SYSTEM
=====================

This document provides instructions for setting up and using the Squat Analysis System.

1. SETUP
--------
a) Create Virtual Environment:
   python -m venv .venv

b) Activate Virtual Environment:
   Windows: .venv\Scripts\activate
   Linux/Mac: source .venv/bin/activate

c) Install Required Packages:
   pip install torch torch_geometric opencv-python mediapipe numpy scipy onnx

2. FILE STRUCTURE
----------------
├── data_preprocessing.py    (Preprocesses training data)
├── gnn_model.py            (Contains GNN model)
├── train.py                (Trains the model)
├── test_video.py           (Tests on videos)
├── convert_to_onnx.py      (Converts to ONNX)
├── dataset/
│   ├── Valid/             (Valid squat sequences)
│   └── Invalid/           (Invalid squat sequences)
└── videos/
    └── test.mp4           (Your test video)

3. USAGE STEPS
-------------
Step 1: Data Preprocessing
        - Run: python data_preprocessing.py
        - Creates preprocessed data files

Step 2: Train Model
        - Run: python train.py
        - Creates best_squat_classifier.pth

Step 3: Test Video
        - Place your video as 'test.mp4'
        - Run: python test_video.py
        - Creates output_test.mp4

Step 4: ONNX Conversion (Optional)
        - Run: python convert_to_onnx.py
        - Creates squat_classifier.onnx

4. INPUT REQUIREMENTS
-------------------
Video:
- MP4 format preferred
- Full body visible
- Good lighting
- Side view recommended
- Contrasting clothing

Dataset:
- Organized in Valid/Invalid folders
- Numbered folders (0-119)
- Multiple .npy files per sequence

5. OUTPUT INTERPRETATION
----------------------
Video Output:
- Green text: Valid Squat
- Red text: Invalid Squat
- Confidence score (0.00-1.00)
- Skeleton overlay on body

6. TROUBLESHOOTING
-----------------
Common Issues:

a) "Model not found":
   - Verify best_squat_classifier.pth exists
   - Check file paths

b) "Video not found":
   - Verify test.mp4 exists
   - Check video file format

c) "CUDA out of memory":
   - Reduce batch size in train.py
   - Use CPU instead of GPU

d) "Package not found":
   - Verify all packages installed
   - Check Python version compatibility

7. BEST PRACTICES
----------------
- Use good lighting
- Keep full body in frame
- Maintain consistent speed
- Wear contrasting clothing
- Face camera from side
- Clear background preferred

8. SYSTEM REQUIREMENTS
--------------------
- Python 3.7 or higher
- CUDA capable GPU (optional)
- Webcam (for live testing)
- 8GB RAM minimum
- 2GB free disk space

9. CONTACT & SUPPORT
------------------
For issues and questions:
1. Check troubleshooting section
2. Verify input requirements
3. Check console output
4. Ensure correct setup

10. VERSION INFO
--------------
Version: 1.0
Last Updated: [Current Date]
Python Version: 3.7+
Key Dependencies:
- PyTorch
- OpenCV
- MediaPipe
- NumPy

Note: Keep all files in the same directory unless specified otherwise.
