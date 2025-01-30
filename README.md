# Face-Recognition-attendance-system-by-insightface-Buffelo_L


Here’s a GitHub repository description for your project:

---

## **Real-Time Face Recognition with InsightFace**

This project implements a real-time face recognition system using the **InsightFace** library, optimized for GPU usage, and designed to handle both **webcam** and **RTSP** camera sources. The system performs **face detection** and **recognition** by comparing live video frames with a database of pre-registered face embeddings. It uses **cosine similarity** to match detected faces with stored embeddings and displays the results in real-time.

### **Features**
- **Real-Time Face Detection**: Detects and recognizes faces in video streams from a webcam or RTSP camera.
- **Face Embeddings**: Utilizes pre-trained face embeddings for recognizing faces. Embeddings are loaded from a file and compared using cosine similarity.
- **GPU Acceleration**: Optimized for GPU usage (CUDA-enabled) to improve processing speed, using the InsightFace library.
- **Multi-Process Architecture**: Captures, processes, and displays video frames using multiple processes for efficiency.
- **Bounding Boxes & Labels**: Displays bounding boxes around detected faces and labels them with recognized names or “Unknown” if no match is found.
- **Timestamp Logging**: Logs recognized faces along with timestamps in the terminal.

### **Requirements**
- Python 3.x
- OpenCV
- InsightFace
- scikit-learn
- numpy
- pandas

### **Installation**
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

Ensure you have a CUDA-enabled GPU for optimal performance.

### **How to Use**
1. Clone the repository.
2. Run the script with the desired camera source (webcam or RTSP stream):
    ```bash
    python face_recognition.py
    ```
3. Choose the camera source (1 for Webcam, 2 for RTSP Stream).
4. The system will process video frames in real-time, display bounding boxes and labels for recognized faces, and log results to the terminal.

### **Note**
- **Embeddings**: Make sure to run the face registration script to generate embeddings before running the face recognition system.
- **Camera Source**: You can specify either a webcam or an RTSP stream URL.

### **Contributing**
Feel free to fork this repository, open issues, and submit pull requests for improvements!

---

This description provides a clear overview of the functionality, installation steps, and usage instructions for your project. Let me know if you'd like to adjust anything!
