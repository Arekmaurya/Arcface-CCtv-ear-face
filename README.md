🧠 CCTVRecognizer: Intelligent CCTV Face & Person Recognition System
🔍 Overview
CCTVRecognizer is a Python-based video processing system that leverages a deep learning–powered MultimodalBiometricSystem to recognize and track individuals in CCTV or recorded video footage.
The system processes frames at a controlled FPS, recognizes people using facial/ear biometrics, tracks individuals across frames, and annotates the video output with bounding boxes, names, and tracking info.

✨ Features
🎥 Real-time video stream or file processing

🧑‍💼 Face and ear-based multimodal biometric recognition

🧠 Deep learning model support (e.g., best_3.pt)

🆔 Persistent tracking with unique person IDs

⚙️ Target FPS control for optimized performance

📝 On-screen overlays: progress, tracking count, frame no.

💾 Optional saving of annotated output video

📦 Requirements
Install dependencies using pip:

pip install opencv-python numpy
🔧 Also include or implement a compatible MultimodalBiometricSystem (e.g., in test.py).

📁 Project Structure
├── cctv_recognizer.py        # 📹 Main CCTV recognizer logic
├── test.py                   # 🧬 MultimodalBiometricSystem implementation
├── best_3.pt                 # 🧠 Pretrained ear recognition model
├── 1.mp4                     # 🎞️ Sample input video
├── output_recognition.mp4    # 💾 Annotated output video (optional)
├── README.md                 # 📄 Project documentation
⚙️ How It Works
🚀 Load the pretrained biometric model

🎞️ Read video frames and skip based on target FPS

🧑‍💼 Recognize persons (face + ear)

🛰️ Track recognized persons with bounding boxes and IDs

🖼️ Annotate frames

💾 Save output or 👀 display in real time

🖥️ Usage
Run the program from terminal:
python cctv_recognizer.py
🔧 Customize input/output video paths and FPS settings in the script.

🧾 Expected Recognition Output Format
Your recognize_person(frame) method should return:
processed_frame, [
  {"name": "Alice", "bbox": (x1, y1, x2, y2)},
  {"name": "Unknown", "bbox": (x1, y1, x2, y2)},
  ...
]
📊 Sample Output Display
Each processed frame will show:

🧾 Name & ID bounding boxes

🖼️ Frame number

🔁 Tracking person count

📈 Progress bar (% of video completed)

🧹 Clean Shutdown
❌ Press Q to exit early

🔓 All video streams are safely released

🪟 OpenCV windows are auto-closed

👤 Author
Harsh Maurya
🧠 Computer Vision & AI/ML Developer

📄 License
🪪 This project is open-source and available under the MIT License.

🤝 Contribute
💡 Found this useful?
⭐ Star the repository, 🍴 fork it, or 🛠️ open a pull request if you'd like to contribute!

