# 🧠 Multi-Task NLP Classifier (Emotion • Hate Speech • Violence)

This project implements a **Multi-Task Learning (MTL)** NLP model using **LSTM** in TensorFlow/Keras.  
The model performs **three text classification tasks simultaneously**:

- **Emotion Detection**
- **Violence Detection**
- **Hate Speech Classification**

A single model shares the core layers (Embedding, LSTM, Dropout, etc.) and produces multiple outputs — one for each task.  
The model is deployed using **Streamlit** for real-time text prediction.

---

## 🚀 Features

✔ Multi-task classification using one model  
✔ Real-time inference with Streamlit  
✔ Text preprocessing (tokenization & stopword removal)  
✔ Individual predictions for Emotion, Violence, and Hate Speech  

---

## 🗂 Datasets Used

| Task      | Dataset Source |
|-----------|----------------|
| Emotion   | Emotions Dataset (Kaggle) |
| Violence  | Gender-Based Violence Tweets (Kaggle) |
| Hate Speech | Hate Speech & Offensive Language (Kaggle) |

---
## 🧠 Download Pretrained Model

The trained model file is too large to include directly in this repository.
You can download it from Google Drive using the link below:

🔗 Model Download:
https://drive.google.com/file/d/12ZRglomEvtg73n5OeuldXqQ7d_Be5_bM/view?usp=sharing

## 🛠 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NLTK**
- **NumPy / Pandas**
- **Streamlit**



