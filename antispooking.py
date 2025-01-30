import cv2
import os
import pandas as pd
from insightface.app import FaceAnalysis
from multiprocessing import Process, Queue, Event
import datetime
from deepface import DeepFace
import math

# Configure GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["INSIGHTFACE_USE_GPU"] = "1"

def load_embeddings(filename="embeddings_o1.pkl"):
    """Load pre-registered face embeddings from file"""
    if os.path.exists(filename):
        return pd.read_pickle(filename)
    raise FileNotFoundError(f"Embeddings file '{filename}' not found.")

def cosine_similarity(a, b):
    """Manual cosine similarity implementation without NumPy"""
    dot_product = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    return dot_product / (norm_a * norm_b)

def process_frame(frame, embeddings, app, similarity_threshold=0.3):
    """Process frame with anti-spoofing and face recognition"""
    try:
        # Maintain aspect ratio while resizing
        h, w = frame.shape[:2]
        new_w = 640
        new_h = int(h * (new_w / w))
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        faces = app.get(resized_frame)

        for face in faces:
            # Manual coordinate scaling
            scale_x = w / new_w
            scale_y = h / new_h
            x1 = max(0, int(face.bbox[0] * scale_x))
            y1 = max(0, int(face.bbox[1] * scale_y))
            x2 = min(w, int(face.bbox[2] * scale_x))
            y2 = min(h, int(face.bbox[3] * scale_y))
            
            if x1 >= x2 or y1 >= y2:
                continue

            # Anti-spoofing check
            try:
                face_region = frame[y1:y2, x1:x2]
                if face_region.size == 0:
                    continue

                # Convert color space
                face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                
                # Spoof detection
                spoof_info = DeepFace.extract_faces(
                    img_path=face_rgb,
                    enforce_detection=False,
                    detector_backend='opencv',
                    anti_spoofing=True
                )[0]
                
                if not spoof_info.get('is_real', True):
                    # Apply blur
                    blurred = cv2.medianBlur(face_region, 99)
                    frame[y1:y2, x1:x2] = blurred
                    print(f"[{datetime.datetime.now()}] Spoof detected!")
                    continue
            except Exception as e:
                print(f"Anti-spoof error: {e}")
                continue

            # Face recognition
            best_match = None
            best_similarity = 0.0
            input_embedding = face.embedding

            for name, saved_embeds in embeddings.items():
                for saved_embed in saved_embeds:
                    current_sim = cosine_similarity(input_embedding, saved_embed)
                    if current_sim > best_similarity:
                        best_similarity = current_sim
                        best_match = name

            # Display results
            label = (f"{best_match} ({best_similarity:.2f})" 
                    if best_match and best_similarity > similarity_threshold 
                    else "Unknown")

            if best_match and best_similarity > similarity_threshold:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Recognized: {best_match} ({best_similarity:.2f})")

            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    except Exception as e:
        print(f"Frame processing error: {e}")
    
    return frame

# Rest of the original code for capture/display processes remains unchanged
# [Keep the capture_frames, process_frames, display_frames, and main functions from previous code]