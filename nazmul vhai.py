import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from multiprocessing import Process, Queue, Event
import datetime

# Ensure GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force use of GPU 0
os.environ["INSIGHTFACE_USE_GPU"] = "1"  # Ensure InsightFace uses GPU

def load_embeddings(filename="embeddings_o1.pkl"):
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

            # Print recognized face and timestamp to the terminal
            if best_match and best_similarity > similarity_threshold:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Recognized: {best_match} with similarity {best_similarity:.2f}")

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

def capture_frames(camera_source, frame_queue, stop_event):
    """Capture frames from the selected camera source."""
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise Exception("Error: Unable to open camera source.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Keep buffer small to reduce lag

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        if not frame_queue.full():
            frame_queue.put(frame)

    cap.release()

def process_frames(frame_queue, processed_frame_queue, embeddings, stop_event):
    """Process frames for face detection and recognition."""
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(320, 320))  # Use GPU (ctx_id=0)

    # Check GPU usage
    gpu_providers = app.models['detection'].session.get_providers()
    print("Available providers:", gpu_providers)
    print("Using GPU:", "CUDAExecutionProvider" in gpu_providers)

    while not stop_event.is_set():
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        processed_frame = process_frame(frame, embeddings, app)
        if not processed_frame_queue.full():
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

def main(camera_source, embeddings):
    """Main function to handle the video stream."""
    frame_queue = Queue(maxsize=10)
    processed_frame_queue = Queue(maxsize=10)
    stop_event = Event()

    # Start the frame capture process
    capture_process = Process(target=capture_frames, args=(camera_source, frame_queue, stop_event), daemon=True)
    capture_process.start()

    # Start the frame processing processes
    processing_processes = []
    for _ in range(4):  # Adjust the number of processes based on your CPU cores
        p = Process(target=process_frames, args=(frame_queue, processed_frame_queue, embeddings, stop_event), daemon=True)
        p.start()
        processing_processes.append(p)

    # Start the frame display in the main thread
    try:
        display_frames(processed_frame_queue, stop_event)
    except KeyboardInterrupt:
        print("Exiting...")
        stop_event.set()

    # Clean up processes
    capture_process.join()
    for p in processing_processes:
        p.join()

if __name__ == "__main__":
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
        main(camera_source, embeddings)
    except Exception as e:
        print(f"Error: {e}")
