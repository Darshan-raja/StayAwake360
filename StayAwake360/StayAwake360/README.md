# 🚗 StayAwake360 – Driver Drowsiness Detection System

StayAwake360 is an AI-powered solution that helps **detect driver drowsiness in real-time** using computer vision techniques. Built with Python, OpenCV, and Dlib, this system monitors facial landmarks to detect early signs of fatigue, ensuring safer roads for everyone.

---

## 🔍 What is StayAwake360?

StayAwake360 is a **driver alertness monitoring system** that uses a webcam to track a driver's eye movements and determine their state:

* ✅ **Active** – Eyes open
* ⚠️ **Drowsy** – Eyes partially closed
* ❌ **Sleeping** – Eyes closed

By continuously analyzing the Eye Aspect Ratio (EAR), the system can raise alerts when drowsiness is detected—potentially preventing accidents before they happen.

---

## 🧠 How It Works

1. **Real-Time Video Capture**
   Captures live video feed using OpenCV from the webcam.

2. **Facial Landmark Detection**
   Uses Dlib’s 68-point facial landmark predictor to identify eye regions.

3. **Eye Aspect Ratio (EAR)**
   Computes EAR to quantify eye openness.

4. **Drowsiness Detection Logic**

   * If EAR remains below a threshold for a defined number of frames, a drowsiness alert is triggered.

5. **Audio Alert (optional)**
   An alarm or notification can be triggered to wake the driver.

---

## 🛠️ Technologies Used

* **Python** 🐍
* **OpenCV** 📸 – Real-time video processing
* **Dlib** 🧠 – Facial landmark detection
* **NumPy** – Numerical operations

---

## 💡 Why StayAwake360?

Drowsiness is one of the leading causes of road accidents. StayAwake360 is a lightweight yet powerful tool aimed at improving driver safety using accessible technology. It’s designed to be:

* 💻 **Easy to run locally**
* ⚡ **Fast and responsive**
* 🔧 **Modular and extendable**

---

## 📂 Project Structure

```
StayAwake360/
│
├── main.py                # Entry point – runs the detection system
├── shape_predictor_68.dat # Dlib’s facial landmark model (download required)
├── utils.py               # EAR calculation and helper functions
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/StayAwake360.git
   cd StayAwake360
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dlib Model**
   Download `shape_predictor_68_face_landmarks.dat` from [Dlib's official source](http://dlib.net/files/) and place it in the project directory.

4. **Run the Application**

   ```bash
   python main.py
   ```

---

## 🛠️ Future Improvements

* Add audio alerts with customizable volume and tone
* Integrate with vehicle systems or mobile apps
* Improve accuracy using deep learning models
* Add support for detecting yawns and head tilts

---

## 🤝 Contributions Welcome

If you have ideas for improving this project, feel free to fork, open issues, or submit pull requests. Let’s make roads safer—together.

---

## 📬 Contact

Have feedback or questions? Reach out on GitHub or connect on [LinkedIn](#).

---

## ⚖️ License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.


