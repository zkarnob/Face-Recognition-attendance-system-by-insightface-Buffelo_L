## 🎭 **Real-Time Face Recognition with InsightFace** 🚀  

Welcome to the **Real-Time Face Recognition System**! This project is designed to detect and recognize faces in **live video streams** 📹 using the **InsightFace** library. Whether you're using a **webcam** or an **RTSP security camera**, this system ensures **fast, accurate, and GPU-accelerated face recognition**! 🔥  

---

### 🚀 **Features at a Glance**
✅ **Real-Time Face Detection & Recognition** – Works with both **webcams** 🎥 and **RTSP streams** 🌍  
✅ **High-Speed Face Matching** – Uses **cosine similarity** to compare live faces with stored embeddings 🧠  
✅ **GPU Acceleration** – Supports **CUDA-enabled GPUs** for **blazing-fast performance** ⚡  
✅ **Multi-Processing for Efficiency** – Uses multiple processes to handle **frame capture, processing, and display** 🔄  
✅ **Smart Bounding Boxes & Labels** – Displays recognized names & marks unknown faces in **real-time** 🏷️  
✅ **Live Logging** – Recognized faces are logged with **timestamps** ⏳ to track entries  

---

### 🔧 **Requirements**
Before you get started, ensure you have the following installed:  

📌 **Python 3.x**  
📌 **OpenCV** – For handling video streams 🖼️  
📌 **InsightFace** – Advanced deep-learning face recognition 🧑‍💻  
📌 **scikit-learn** – For calculating **cosine similarity** 📊  
📌 **NumPy & Pandas** – For efficient data handling 📑  

🔹 **Install all dependencies with:**
```bash
pip install -r requirements.txt
```
💡 **Tip:** Make sure your **GPU is enabled** for the best performance!  

---

### 🎬 **How to Run**
1️⃣ **Clone this repo** 🛠️  
```bash
git clone https://github.com/yourusername/real-time-face-recognition.git
cd real-time-face-recognition
```
  
2️⃣ **Run the script & choose the camera source** 🔄  
```bash
python face_recognition.py
```
  
3️⃣ **Choose your input method:**  
   - Press `1` for **Webcam** 🎥  
   - Press `2` for **RTSP Stream** 🌐  

4️⃣ **Watch the magic happen!** 🎩 The system will **detect faces, match them with known embeddings, and display live results.**  

---

### 📌 **Important Notes**
⚠️ **Ensure embeddings are pre-generated** before running recognition.  
⚠️ **Works best with a CUDA-supported GPU** for smooth processing.  
⚠️ **RTSP streams require a valid network camera setup.**  

---

### 🤝 **Contributing**
We love contributions! 🎉 Want to add new features or fix bugs? **Fork this repo, create a branch, and submit a PR!** 🔥  

📢 **Join the Discussion!** Have questions or suggestions? Open an **Issue** or start a discussion! 🗣️  

---

### 🎯 **Future Enhancements**
🚀 **Add support for multiple cameras** 🎥  
🚀 **Improve recognition accuracy with fine-tuned models** 🧠  
🚀 **Web dashboard for live tracking & analytics** 📊  

💡 **Got Ideas?** Let us know how we can make this even better!  

🔗 **Stay Connected & Star this Repo!** ⭐  

Happy Coding! 🚀💻  

---

