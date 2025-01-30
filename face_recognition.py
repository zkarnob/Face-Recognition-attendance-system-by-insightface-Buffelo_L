import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from multiprocessing import Process, Queue, Event
import datetime

# Load embeddings from file
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

            # Normalize the input embedding
            input_embedding = input_embedding / np.linalg.norm(input_embedding)

            best_match = None
            best_similarity = 0

            # Compare embeddings with saved ones
            for name, saved_embeddings in embeddings.items():
                saved_embeddings_array = np.array(saved_embeddings)

                # Normalize saved embeddings
                saved_embeddings_array = saved_embeddings_array / np.linalg.norm(saved_embeddings_array, axis=1, keepdims=True)

                similarities = cosine_similarity(input_embedding, saved_embeddings_array).flatten()
                max_similarity = similarities.max()

                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_match = name

            label = f"{best_match} ({best_similarity:.2f})" if best_match and best_similarity > similarity_threshold else "Unknown"

            # Print recognized face and timestamp
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
