<!--
-->

# 🤟 ISL Sign Language Recognition System

> " This app is a real-time Indian Sign Language recognition system that converts hand signs into spoken sentences using AI

---

## 👤 Project Information

| Field | Details                              |
|---|--------------------------------------|
| **Student Name** | Jigar J langaliya                    |
| **Enrollment Number** | 246490316114                         |
| **College** | Sir Bhavsihnji Polytechnoc Institute |
| **Training Center** | UFOS Training Center, Bhavnagar      |
| **Program** | AI Internship 2026 — 15 Days         |
| **Mentor / Guide** | Aditya Dodiya                        |

---

## 📌 Project Overview

This project is an AI powered Indian Sign Language (ISL) Translator that enables real time communication between sign language users and non sign language users. Using computer vision and machine learning, the system detects hand gestures through a webcam and accurately recognizes ISL signs. The recognized signs are converted into meaningful words, combined into sentences, and then transformed into speech using text to speech technology.

 With additional features such as blink based speech triggering.

This project demonstrates a complete **MLOps lifecycle** — from data collection through model training, evaluation, quality testing, and deployment to a live, publicly accessible web application. It is not just a trained model; it is a **shipped, working product**.

---

## 🔄 MLOps Lifecycle Map

> Most student AI projects stop at "I trained a model." This project goes further — it covers the complete production lifecycle that real ML engineering teams follow.

| MLOps Stage | What It Means | File / Artifact In This Project |
|---|---|---|
| **1. Data Collection** | Gathering raw training data from a real source | `isl_data_collector.py`, `new_signs_collector.py` |
| **2. Data Merging / Versioning** | Combining data from multiple contributors into one clean dataset | `merge_datasets.py`, `merge_new_signs.py` |
| **3. Model Training** | Teaching an algorithm to recognize patterns in the data | `train_model.py` |
| **4. Model Evaluation** | Measuring how good the model actually is | `accuracy_compare.py`, confusion matrix, per-sign accuracy |
| **5. Model Versioning** | Keeping previous model versions safe before replacing them | `isl_final_fully_model.pkl`, backup model files |
| **6. Quality Assurance** | Systematic testing — not random poking around | Day 11 QA Testing Booklet (accuracy, stress, cross, edge-case testing) |
| **7. Monitoring / Logging** | Tracking real usage after the model is running | Session statistics tracked in-app |
| **8. Deployment** | Making the model accessible to real users, anywhere | Streamlit + streamlit-webrtc + Hugging Face Spaces |
| **9. Feedback Loop** | Using real-world feedback to improve the model after launch | Multi-person data re-collection → retrain cycle |

---

## ✨ Features

- ✅ Real-time hand sign detection using a live webcam feed, directly in the browser
- ✅ Recognizes **13 ISL signs** (10 foundational + 3 expanded: ILY, OK, Eat)
- ✅ Automatic sentence builder — signs become words as you sign, no manual capture needed
- ✅ Confidence-based prediction (minimum 70% confidence before a word is accepted)
- ✅ Stability filtering — a sign must hold steady for 3 consecutive frames before being trusted
- ✅ Cooldown timer (1.5s) — prevents the same sign from being captured twice in a row
- ✅ **Eye-blink trigger** — blink 3 times to speak the sentence aloud, completely hands-free
- ✅ **Smart alert system** — if you blink 3 times with an empty sentence, the app speaks a spoken warning instead of silently doing nothing
- ✅ Offline text-to-speech output (pyttsx3 — no internet required for speech)
- ✅ Translation layer — internal model labels (e.g. `open_palm`) convert to natural human language (`Hello`)
- ✅ Live on-screen overlays: blink counter, current predicted sign, and building sentence — all drawn directly on the video feed
- ✅ Animated, color-coded web UI (blink counter, sentence panel, speaking indicator, alert banner)
- ✅ Manual controls: Backspace, Clear All, and Reset Blinks buttons
- ✅ Thread-safe architecture — speech runs in a background thread so the camera never freezes
- ✅ Trained on hand-sign data from **✏️ [WRITE: number] different people** for improved real-world accuracy
- ✏️ [WRITE: any additional feature you personally added]

---

## 🛠️ Tech Stack

| Tool / Library   | Purpose In This Project |
|------------------|---|
| Python 3.8       | Core programming language |
| Streamlit        | Turns the Python script into an interactive web application |
| streamlit-webrtc | Streams continuous live video into the browser (not just single photos) |
| OpenCV           | Frame processing, flipping, drawing overlays (blink counter, sign text, sentence) |
| MediaPipe        | Hand landmark detection (21 points) + Face Mesh (eye blink detection) |
| scikit-learn     | RandomForest model training and evaluation |
| pandas           | Dataset loading, merging, and shuffling |
| NumPy            | Numerical feature array computation |
| pyttsx3          | Offline text-to-speech engine (local testing) |
| joblib           | Saving and loading the trained model |
| threading        | Runs speech in the background so video never freezes |

---

## 📦 Installation

**Step 1 — Verify Python version**
```bash
python --version
```

**Step 2 — Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Step 3 — Install all required libraries**
```bash
pip install streamlit
pip install streamlit-webrtc
pip install opencv-python
pip install mediapipe
pip install joblib
pip install numpy
pip install scikit-learn
pip install pandas
pip install pyttsx3
pip install av
```

**Step 4 — Verify installation**
```python
import streamlit, cv2, mediapipe, sklearn, pyttsx3
print("All libraries installed successfully.")
```

---

## ▶️ How To Run Locally

**Required files in the project folder:**
```
app.py                       ← main application (run this)
blink_detector.py            ← eye blink detection module
isl_final_fully_model.pkl    ← trained model
```

**Run the application:**
```bash
streamlit run app.py
```

This opens a local browser tab. Click **START** under the video panel and allow camera access when prompted.

---

## 🎮 Controls

| Input | Action |
|---|---|
| Show a hand sign | Detected and added to the sentence automatically |
| **Blink × 3 (with a sentence built)** | Speaks the sentence aloud, then clears it |
| **Blink × 3 (with an empty sentence)** | Speaks a warning: *"No sentence to speak. Please make a sign first."* |
| **⌫ Backspace** | Removes the last word from the sentence |
| **🗑️ Clear All** | Clears the entire sentence |
| **🔄 Reset Blinks** | Manually resets the blink counter to 0/3 |

---

## 🤟 Signs Supported (13 Total)

| # | Sign Label | Spoken / Displayed As | Hand Shape Description |
|---|---|---|---|
| 1 | open_palm | Hello | All 5 fingers extended, palm facing camera |
| 2 | thumbs_up | Good | Closed fist, thumb pointing up |
| 3 | thumbs_down | No | Closed fist, thumb pointing down |
| 4 | pointing | Look | Only the index finger extended |
| 5 | fist | Stop | All fingers curled into a closed fist |
| 6 | thank_you | Thank You | Flat hand moves from chin outward |
| 7 | sorry | Sorry | Closed fist, circular motion on chest |
| 8 | please | Please | Open hand, circular motion on chest |
| 9 | unhappy | Sad | Index finger traces a line down the cheek |
| 10 | water | Water | "W" hand shape taps the chin |
| 11 | ily | I Love You | Thumb, index, and pinky extended |
| 12 | ok | OK | Thumb and index form a circle |
| 13 | eat | Eat | All fingertips pinched together |

---

## 📁 Project File Structure

```
isl-project/
│
├── app.py                       ← Main Streamlit application (run this)
├── blink_detector.py            ← Eye blink detection module (EAR formula)
├── train_model.py               ← Script to train the model
├── isl_data_collector.py        ← Collect training data (10 original signs)
├── new_signs_collector.py       ← Collect new signs data (ILY, OK, Eat)
├── merge_datasets.py            ← Merge multiple classmate CSVs
├── merge_new_signs.py           ← Merge new signs into main dataset
├── accuracy_compare.py          ← Compare model accuracy before/after merging
│
├── isl_final_fully_model.pkl    ← Final trained RandomForest model
│
├── isl_dataset.csv              ← Original single-person dataset
├── isl_dataset_combined.csv     ← Multi-person combined dataset
├── new_signs.csv                ← New signs raw data
├── isl_dataset_final.csv        ← Final complete training dataset
│
└── README.md                    ← This file
```

---

## 📊 Model Details

| Metric | Value                                              |
|---|----------------------------------------------------|
| Algorithm | RandomForest Classifier                            |
| Number of Trees | 100                                                |
| Features Per Sample | 45 (42 relative coordinates + 3 distance features) |
| Number of Classes (Signs) | 13                                                 |
| Stability Threshold | 3 consecutive frames                               |
| Confidence Threshold | 70% minimum                                        |
| Capture Cooldown | 1.5 seconds                                        |
| Train/Test Split | 80% / 20%                                          |
| Number of Data Contributors | 8 other peoples                                    |
| Total Training Samples | 11015                                              |
| Model Accuracy | 89.86 %                                            |

---

## 🔬 How The System Works

```
1.  Camera captures a live video frame inside the browser
2.  MediaPipe detects 21 hand landmarks + face mesh landmarks
3.  Hand coordinates are normalized relative to the wrist
4.  3 additional distance features are calculated
5.  45 total features are passed to the RandomForest model
6.  100 decision trees vote on the most likely sign
7.  The prediction must remain stable for 3 consecutive frames
8.  The prediction must have 70%+ confidence to be accepted
9.  A 1.5-second cooldown prevents duplicate captures
10. The recognized word is translated and added to the sentence
11. Eye Aspect Ratio (EAR) from Face Mesh detects each blink
12. After 3 blinks: if a sentence exists, it is spoken aloud
                     if no sentence exists, a spoken alert plays instead
13. Speech runs in a background thread so the camera feed never freezes
14. The sentence clears automatically and the blink counter resets
```

---

## 🧵 Why Threading Matters Here

Text-to-speech (`pyttsx3`) takes 2–5 seconds to finish speaking a sentence. If this ran directly inside the video processing loop, the camera would freeze for the entire duration of the speech. By running `speak_sentence_threaded()` in its own background thread, the camera keeps capturing and processing frames continuously while the sentence is being spoken in parallel.

---

## ➕ How To Add More Signs

**Step 1 — Collect data for the new sign**
```bash
python new_signs_collector.py
```

**Step 2 — Merge the new data with the existing dataset**
```bash
python merge_new_signs.py
```

**Step 3 — Update `SIGN_LABELS` in `app.py`**
```python
SIGN_LABELS = {
    'your_new_sign': 'Display Text',
}
```

**Step 4 — Retrain the model**
```bash
python train_model.py
```

---

## 📈 Dataset Summary

| Dataset File | Rows     | Description                                |
|---|----------|--------------------------------------------|
| isl_dataset.csv | 701      | Original data — personal hand samples only |
| isl_dataset_combined.csv | 4481     | Combined with 8 classmates                 |
| new_signs.csv | 313      | Raw data for ILY, OK, and Eat              |
| isl_dataset_final.csv | 4793     | Final merged dataset used for training     |

**Data contributors:** Total 9 Peoples - 8 Classmates & Myself  

---

## 📋 Quality Assurance Testing Summary

*(Completed during the Day 11 QA Testing Booklet)*

| Test Round | Result              |
|---|---------------------|
| Accuracy Testing — overall average (13 signs) | 89.86 %             |
| Best performing sign | Thumb_Down + 95.1%  |
| Weakest performing sign | I Love You  + 83,5 % |
| Stress Tests passed | 8/ 10               |
| Cross-Testing generalization grade | B                   |
| Edge Cases handled correctly | 5 / 7               |

---

## 🌐 Deployment

This application is also deployed as a live, publicly accessible web app using **Hugging Face Spaces**. The deployed version uses:
- **gTTS** instead of pyttsx3 (generates an audio file sent to the browser, since cloud servers have no physical speaker)
- **Twilio TURN servers** instead of plain STUN (required because cloud server networks are behind strict firewalls that block direct peer-to-peer video connections)

**Live link:** https://huggingface.co/spaces/JigarLangalia/isl_sign_language

---

## 🧠 What I Learned In This Internship

The Most important concept that i learnt by this internship was about Deployment Phase of he MLOPs, How To Deploy your project from your laptop to global service.The Hardest bug was to fix any sign that are being detected correct in starting but getting Wrong in the ending, also the bug or error of saving the data in the streamlit session was hard to solve. Taking my model from my local personal laptop to a web-server was a joyful moment that my fist internship project was done Very Easily.  

>"*I Learned So much things that were helpful for me to bring out the coding logic & different problem solving strategies*"  


---

## 🙏 Acknowledgements

- **Mentor / Guide:** Aditya Dodiya
- **Training Center:** UFOS Training Center, Bhavnagar
- **ISL Reference:** Indian Sign Language Research and Training Centre — islrtc.nic.in
- **Core Libraries:** OpenCV, MediaPipe (Google), scikit-learn, Streamlit, pyttsx3
- **Data Contributors:** vedant, Kavya, Harshit, Bhakti, vidhi, Jeet, Tirth, Dhruvi

---

## 📞 Contact

| Field | Details                      |
|---|------------------------------|
| **Name** | Jigar J Langaliya            |
| **Email** | jigarlangaliya108@gmail.com  |
| **College** | Sir Bhavsihnji Polytechnic Institute                            |

---

<p align="center"><i>Built at UFOS Training Center — AI Internship 2026 — Bhavnagar, Gujarat</i></p>
