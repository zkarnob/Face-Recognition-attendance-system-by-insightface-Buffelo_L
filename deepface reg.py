import os
import cv2
import pandas as pd
from deepface import DeepFace

# Encode images using DeepFace (arcface model)
def encode_faces_deepface(dataset_path):
    embeddings = {}
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            embeddings[person_name] = []
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    # Load image
                    img = cv2.imread(image_path)

                    # Use DeepFace to get embeddings using ArcFace
                    embedding = DeepFace.represent(img_path=image_path, model_name="ArcFace", enforce_detection=False)

                    embeddings[person_name].append(embedding[0]['embedding'])
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    return embeddings

# Save embeddings to a file
def save_embeddings(embeddings, filename="embeddingsdeep.pkl"):
    pd.to_pickle(embeddings, filename)

if __name__ == "__main__":
    dataset_path = "./50mp jpg dataset"  # Path to the dataset
    embeddings = encode_faces_deepface(dataset_path)  # Generate embeddings using DeepFace
    save_embeddings(embeddings)  # Save embeddings to a file
    print("Face registration complete! Embeddings saved to embeddings.pkl")
