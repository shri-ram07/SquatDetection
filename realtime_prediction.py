import cv2
import numpy as np
import torch
import mediapipe as mp
from gnn_model import SquatGNN, create_graph_from_sequence


class SquatVideoTester:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading model...")
        self.model = SquatGNN(input_dim=132, hidden_dim=64, output_dim=2).to(self.device)
        self.model.load_state_dict(torch.load('best_squat_classifier.pth', map_location=self.device))
        self.model.eval()

        # Buffer for frames
        self.frame_buffer = []
        self.buffer_size = 100

    def extract_keypoints(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            # Extract keypoints
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # Draw skeleton
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            return np.array(keypoints)
        return None

    def predict_squat(self, keypoints_sequence):
        # Create graph from sequence
        graph = create_graph_from_sequence(keypoints_sequence)
        graph = graph.to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model.forward_data(graph)
            probabilities = torch.exp(output)
            prediction = torch.argmax(output).item()
            confidence = probabilities[0][prediction].item()

        return "Valid Squat" if prediction == 1 else "Invalid Squat", confidence

    def process_video(self, video_path):
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Create output video writer
        output_path = 'output_test.mp4'
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (frame_width, frame_height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}", end='\r')

            # Get keypoints
            keypoints = self.extract_keypoints(frame)

            if keypoints is not None:
                self.frame_buffer.append(keypoints)

                # Make prediction when buffer is full
                if len(self.frame_buffer) >= self.buffer_size:
                    status, confidence = self.predict_squat(np.array(self.frame_buffer))

                    # Draw prediction
                    color = (0, 255, 0) if "Valid" in status else (0, 0, 255)
                    cv2.putText(frame, f"{status} ({confidence:.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 2)

                    # Remove oldest frame
                    self.frame_buffer.pop(0)

            # Show frame
            cv2.imshow('Squat Analysis', frame)
            out.write(frame)

            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.pose.close()

        print(f"\nProcessing complete! Output saved as: {output_path}")


if __name__ == "__main__":
    # Create tester instance
    tester = SquatVideoTester()

    # Process video
    video_path = "test.mp4"  # Your video file
    tester.process_video(video_path)