import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis


def load_embeddings(filename="embeddings.pkl"):
    """Load embeddings from a file."""
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    else:
        raise FileNotFoundError(f"Embeddings file '{filename}' not found. Please run the face registration script first.")


def process_frame(frame, embeddings, app, similarity_threshold=0.5):
    """Process a single frame for face detection and recognition."""
    try:
        resized_frame = cv2.resize(frame, (640, 360))
        faces = app.get(resized_frame)

        for face in faces:
            input_embedding = face.embedding.reshape(1, -1)
            best_match = None
            best_similarity = 0

            # Compare embeddings
            for name, saved_embeddings in embeddings.items():
                saved_embeddings_array = np.array(saved_embeddings)
                similarities = cosine_similarity(input_embedding, saved_embeddings_array).flatten()
                max_similarity = similarities.max()
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_match = name

            # Determine recognition result
            label = f"{best_match} ({best_similarity:.2f})" if best_match and best_similarity > similarity_threshold else "Unknown"
            print(f"Detected: {label}")

            # Draw bounding box and label
            bbox = face.bbox.astype(int)
            scale_x = frame.shape[1] / resized_frame.shape[1]
            scale_y = frame.shape[0] / resized_frame.shape[0]
            bbox = [int(bbox[0] * scale_x), int(bbox[1] * scale_y), int(bbox[2] * scale_x), int(bbox[3] * scale_y)]

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)
    except Exception as e:
        print(f"Error during face detection: {e}")
    return frame


def main(camera_source, embeddings, app):
    """Main function to handle the video stream."""
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise Exception("Error: Unable to open camera source.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' to exit the live feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Retrying...")
            continue

        # Process the frame for face detection and recognition
        processed_frame = process_frame(frame, embeddings, app)

        # Display the frame
        cv2.imshow("Live Face Recognition", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # User input for camera selection
    print("Choose camera source:")
    print("1. Webcam")
    print("2. RTSP Stream")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        camera_source = 0  # Webcam
    elif choice == "2":
        camera_source = "rtsp://rndteam:rnddev1234%@192.168.6.76/media/video1&tcp"  # RTSP URL
    else:
        print("Invalid choice. Exiting.")
        exit(1)

    try:
        embeddings = load_embeddings()
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0, det_size=(640, 640))  # Use GPU (ctx_id=0) or CPU (ctx_id=-1)
        main(camera_source, embeddings, app)
    except Exception as e:
        print(f"Error: {e}")
