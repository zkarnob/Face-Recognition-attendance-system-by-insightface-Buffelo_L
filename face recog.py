import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import multiprocessing as mp
import threading
import time


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


def capture_frames(camera_source, frame_queue, stop_event, frame_skip=2):
    """Capture frames from the selected camera source."""
    cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG) if isinstance(camera_source, str) else cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise Exception("Error: Unable to open camera source.")
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames to reduce lag
            continue

        if frame_queue.full():
            frame_queue.get()  # Drop the oldest frame
        frame_queue.put(frame)

    cap.release()


def recognize_faces_worker(frame_queue, processed_frame_queue, embeddings, app, stop_event):
    """Recognize faces in frames."""
    while not stop_event.is_set():
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        processed_frame = process_frame(frame, embeddings, app)
        if processed_frame_queue.full():
            processed_frame_queue.get()  # Drop the oldest processed frame
        processed_frame_queue.put(processed_frame)


def display_frames(processed_frame_queue, stop_event):
    """Display processed frames."""
    while not stop_event.is_set():
        if processed_frame_queue.empty():
            continue
        frame = processed_frame_queue.get()
        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    cv2.destroyAllWindows()


def main(camera_source, embeddings, app):
    """Main function to handle the video stream."""
    frame_queue = mp.Queue(maxsize=5)
    processed_frame_queue = mp.Queue(maxsize=5)
    stop_event = mp.Event()

    # Start the frame capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(camera_source, frame_queue, stop_event), daemon=True)
    capture_thread.start()

    # Start face recognition processes
    processes = []
    for _ in range(mp.cpu_count()):  # One process per CPU core
        p = mp.Process(target=recognize_faces_worker, args=(frame_queue, processed_frame_queue, embeddings, app, stop_event), daemon=True)
        processes.append(p)
        p.start()

    # Start display thread
    display_thread = threading.Thread(target=display_frames, args=(processed_frame_queue, stop_event), daemon=True)
    display_thread.start()

    # Wait for all threads and processes to complete
    try:
        display_thread.join()
    except KeyboardInterrupt:
        print("Exiting...")
        stop_event.set()

    # Clean up
    for p in processes:
        p.join()
    capture_thread.join()


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
