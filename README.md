🧠 AI-Based Dyslexia Detection Using Deep Learning
👤 Author: Madhavan A Ramanujan

B.Tech CSE (AI & Robotics), VIT Chennai
GitHub: Maddy152006




📝 Abstract
This project presents an AI-powered Dyslexia Detection System that identifies dyslexic handwriting using deep learning and OCR (Optical Character Recognition).
The model analyzes handwritten text images and classifies individual characters into three categories — Normal, Reversal, and Corrected — using a trained CNN model named safe_glyphnet_best.keras.
The main goal is to assist in early dyslexia screening by providing a web-based application that can automatically detect patterns commonly associated with dyslexic handwriting.




🔍 Overview of the Project
Objective:
To detect handwriting patterns indicative of dyslexia using computer vision and AI.


Core Components:
CNN-based character-level classifier
Flask-based web app for interactive image upload and visualization
PyTesseract OCR for extracting text
Automated bounding-box annotation for reversed or uncertain characters




Output:
“✅ No Dyslexia Detected”
or “⚠️ Possible Dyslexia Detected” with highlighted problem areas.






⚙️ System Architecture
        +---------------------------+
        |  Input Handwritten Image  |
        +-------------+-------------+
                      |
                      v
         +--------------------------+
         |   OCR (Tesseract Engine) |
         +-------------+------------+
                       |
                       v
        +-----------------------------+
        |   Preprocessing & Cropping  |
        +-------------+---------------+
                      |
                      v
         +----------------------------+
         |   CNN (safe_glyphnet_best) |
         |   Predicts: Normal/Reversal|
         +-------------+--------------+
                       |
                       v
        +----------------------------+
        |   Dyslexia Decision Logic  |
        |  (counts reversals, scores)|
        +-------------+--------------+
                      |
                      v
       +-----------------------------------+
       | Flask Web Interface (app.py)      |
       | Result, Annotated Image, Summary  |
       +-----------------------------------+






🧩 Methodology
Dataset Preparation
Dataset structure:

dataset/
├── Train/
│   ├── Normal/
│   ├── Reversal/
│   └── Corrected/
└── Test/
    ├── Normal/
    ├── Reversal/
    └── Corrected/


Each folder contains grayscale character images extracted from handwritten text.




Model Training
A Convolutional Neural Network (CNN) was trained on these character classes.
Model: safe_glyphnet_best.keras
Input size: 96x96x1
Optimizer: Adam
Loss: Categorical Cross-Entropy



Web Application
Flask app (app.py) allows users to upload images.
PyTesseract extracts text and bounding boxes.
The CNN model analyzes cropped characters.
The final result is rendered on an HTML page with annotated bounding boxes.




Decision Rule
If >40% of characters are reversed or uncertain → flag as “Possible Dyslexia Detected”.
Otherwise → “No Dyslexia Detected”.





🧮 Evaluation Metrics
Metric	Description
Accuracy	Percentage of correctly classified characters
Precision	Correct positive predictions (Reversal/Normal)
Recall	Sensitivity to dyslexic reversals
F1 Score	Combined precision and recall
Confusion Matrix	Distribution of correct/incorrect predictions





📊 Results
Metric	Value
Training Accuracy	~91%
Validation Accuracy	~88%
Input Size	96×96 grayscale
Epochs	8
Learning Rate	1e-4
🧠 Model Output Example
Uploaded Image	Annotated Output	Result

	⚠️ Possible Dyslexia Detected
Red boxes → Reversed characters
Blue boxes → Auto-corrected characters





🚀 How to Run Locally
git clone https://github.com/Maddy152006/dyslexia-detection-ai.git
cd dyslexia-detection-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py




Then open your browser and visit:
👉 http://127.0.0.1:5000




🧩 Project Folder Structure
project_ai/
│
├── app.py                        # Flask web app
├── dataset/                      # Training + Test data (excluded from GitHub)
│   ├── Train/
│   └── Test/
├── notebook/                     # Contains models and Jupyter notebooks
│   ├── safe_glyphnet_best.keras
│   ├── train_clean_glyphs.ipynb
│   └── eval_model.ipynb
├── static/                       # Annotated output images
├── requirements.txt
└── README.md




🧾 Conclusion
This project demonstrates how AI and OCR can be combined to identify patterns in handwriting associated with dyslexia.
While the model is currently trained on limited samples, its architecture shows strong potential for real-world screening tools when expanded with larger, more diverse datasets.




🧠 Future Enhancements

Integrate live camera feed for real-time detection
Expand dataset for improved generalization
Build mobile app version for educators & therapists
Add word-level analysis using NLP



📬 Contact
Madhavan A Ramanujan
📧 madhavanaramanujan@gmail.com


💻 GitHub Profile
🌟 If you find this project helpful, please star ⭐ the repo on GitHub!
