# 🧠 Personality Prediction Using Video-Based VGG-Face Transfer Learning

This project is a deep learning system to predict personality traits based on facial video data using the Big Five Personality model (OCEAN). It utilizes **transfer learning with VGG-Face**, **MTCNN** for face detection, and a custom LSTM architecture to extract temporal facial features and perform personality regression.

## 📌 Project Overview

- **Title**: Personality Prediction Using Video-Based VGG-Face Transfer Learning Model  
- **Author**: Achmad Nurs Syururi Arifin  
- **Institution**: Universitas Negeri Surabaya  
- **Degree**: Bachelor of Applied Computer Science  
- **Year**: 2025  

## 🎯 Objective

To build a personality prediction system that can automatically classify personality traits using only facial expressions from video clips, reducing dependency on traditional self-reported questionnaires.

## 🧪 Methodology

1. **Face Detection**: MTCNN  
2. **Feature Extraction**: VGG-Face pretrained model  
3. **Model Architecture**:
   - VGG-Face embeddings
   - 2 LSTM layers
   - Dense + Dropout layers
4. **Regression Output**: Predict 5 continuous scores for:
   - Openness
   - Conscientiousness
   - Extraversion
   - Agreeableness
   - Neuroticism

## 📊 Dataset

- **Name**: ChaLearn LAP CVPR 2017  
- **Size**: ~10,000 annotated video clips  
- **Duration**: ~15 seconds each  
- **Classes**: Big Five traits (OCEAN) scored per video  

## 📈 Results

| Phase     | Accuracy (converted from MAE) |
|-----------|-------------------------------|
| Training  | 91.75%                        |
| Validation| 90.39%                        |
| Testing   | 90.28%                        |

## 💻 Tech Stack

- Python
- TensorFlow & Keras
- OpenCV
- Scikit-learn
- Flask (Web deployment)
- MTCNN
- VGG-Face
- Pandas, Matplotlib

## 🚀 Features

- Upload and analyze video directly via web interface
- Real-time preview of face detection
- Downloadable personality results in **PDF** and **CSV**
- Fully offline compatible system

## 📷 Screenshots

| Upload Video Page | Prediction Result |
|-------------------|------------------|
| ![upload](assets/upload_page.png) | ![result](assets/result_page.png) |

## 📂 Folder Structure
project/
│
├── app/ # Flask app
│ ├── static/
│ ├── templates/
│ ├── model/ # Saved model
│ └── app.py
├── data/ # Dataset samples
├── notebook/ # Jupyter experiments
├── requirements.txt
└── README.md

## 📚 References

This project is inspired and adapted from the following scientific work:

- Ayoub Ouarka, Tarek Ait Baha, Youssef Es-Saady, Mohamed El Hajji.  
  *A Deep Multimodal Fusion Method for Personality Traits Prediction*.  
  **Multimedia Tools and Applications (2024)**.  
  [DOI: 10.1007/s11042-024-20356-y](https://doi.org/10.1007/s11042-024-20356-y)
