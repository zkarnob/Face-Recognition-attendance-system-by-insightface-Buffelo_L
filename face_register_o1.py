import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import time

# Initialize InsightFace (buffalo_l model) for face recognition using GPU
def initialize_insightface():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))  # Use GPU (ctx_id=0)
    return app

# Process each person's images in parallel using multiprocessing
def process_person_images(person_path, app):
    embeddings = []
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        if image_path.endswith(('.png', '.jpg', '.jpeg')):  # Process PNG, JPG, and JPEG
            img = cv2.imread(image_path)  # Read image using OpenCV
            if img is not None:
                faces = app.get(img)
                for face in faces:
                    embeddings.append(face.embedding)
    return embeddings

# Function to encode faces from the dataset
def encode_faces_insightface(dataset_path, app):
    embeddings = {}
    
    # Get all person directories in the dataset
    person_dirs = [os.path.join(dataset_path, person_name) for person_name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, person_name))]
    
    # Using multiprocessing to process multiple people in parallel
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(process_person_images, person_path, app) for person_path in person_dirs]
        
        for idx, future in enumerate(futures):
            person_name = os.path.basename(person_dirs[idx])
            try:
                embeddings[person_name] = future.result()
            except Exception as e:
                print(f"Error processing {person_name}: {e}")

    return embeddings

# Normalize embeddings (L2 normalization)
def normalize_embeddings(embeddings):
    for person_name in embeddings:
        embeddings[person_name] = normalize(np.array(embeddings[person_name]))
    return embeddings

# Save embeddings to a pickle file
def save_embeddings(embeddings, filename="embeddings_o1m.pkl"):
    pd.to_pickle(embeddings, filename)

# Main function to run face registration
def main(dataset_path):
    start_time = time.time()
    app = initialize_insightface()  # Initialize InsightFace (buffalo_l model)
    
    # Process and encode faces from dataset
    embeddings = encode_faces_insightface(dataset_path, app)
    
    # Normalize the embeddings
    embeddings = normalize_embeddings(embeddings)
    
    # Save embeddings to file
    save_embeddings(embeddings)
    
    print(f"Face registration complete! Embeddings saved to embeddings.pkl")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    dataset_path = "./50mp jpg dataset"  # Path to your dataset
    main(dataset_path)
