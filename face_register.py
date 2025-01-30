import os
import cv2
import pandas as pd
from deepface import DeepFace
from insightface.app import FaceAnalysis

# Initialize InsightFace
def initialize_insightface():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# Encode images using InsightFace
def encode_faces_insightface(dataset_path, app):
    embeddings = {}
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            embeddings[person_name] = []
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    img = cv2.imread(image_path)
                    faces = app.get(img)
                    for face in faces:
                        embeddings[person_name].append(face.embedding)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    return embeddings

# Save embeddings to a file
def save_embeddings(embeddings, filename="embeddings.pkl"):
    pd.to_pickle(embeddings, filename)

if __name__ == "__main__":
    dataset_path = "./dataset"  # Path to the dataset
    app = initialize_insightface()  # Initialize the face analysis model
    embeddings = encode_faces_insightface(dataset_path, app)  # Generate embeddings
    save_embeddings(embeddings)  # Save embeddings to a file
    print("Face registration complete! Embeddings saved to embeddings.pkl")
