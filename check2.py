import cv2
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
from multiprocessing import Process, Queue, Event
import datetime
import torch  # To verify GPU usage
import torch

import torch
import insightface
from deepface import DeepFace

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"InsightFace version: {insightface.__version__}")
print(f"DeepFace version: {DeepFace.__version__}")
