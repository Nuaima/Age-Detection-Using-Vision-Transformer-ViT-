# 🖼️ Age Detection Using Vision Transformer (ViT)

### 📌 Overview
This project implements **Age Detection** from facial images using the **Vision Transformer (ViT)** model.  
The workflow includes **data preprocessing, oversampling, training, evaluation, and deployment** of the model on **Hugging Face Hub**.  
It uses the **Faces Age Detection Dataset** and classifies images into three age groups:  
- **YOUNG**  
- **MIDDLE**  
- **OLD**

---

### 🛠️ Technologies Used
- **Programming Language:** Python 3  
- **Libraries & Frameworks:**
  - [PyTorch](https://pytorch.org/) → Model training and deep learning framework  
  - [Transformers](https://huggingface.co/transformers/) → Pretrained ViT models and Trainer API  
  - [Datasets](https://huggingface.co/docs/datasets) → Handling image datasets  
  - [scikit-learn](https://scikit-learn.org/stable/) → Evaluation metrics (Accuracy, F1, Confusion Matrix)  
  - [imbalanced-learn](https://imbalanced-learn.org/stable/) → Oversampling minority classes  
  - [Matplotlib & Seaborn](https://matplotlib.org/) → Data visualization  
  - [PIL](https://pillow.readthedocs.io/en/stable/) → Image preprocessing  
  - [MLflow](https://mlflow.org/) → Experiment tracking  
  - [Hugging Face Hub](https://huggingface.co/) → Model hosting and deployment  

---

### 📂 Dataset
- **Source:** [Faces Age Detection Dataset (Kaggle)](https://www.kaggle.com/datasets)  
- **Location in Kaggle:** `/kaggle/input/faces-age-detection-dataset/train.csv`  
- **Structure:**  
  - Images of faces with corresponding labels (`YOUNG`, `MIDDLE`, `OLD`)  
  - CSV file contains image filenames and class labels  
  - Images stored in `/kaggle/input/faces-age-detection-dataset/Train/`  
- **Split:** ~90% Training | 10% Testing  

---

### 📁 Repository Structure
📦 Age-Detection-ViT/
├── 📄 train_notebook.ipynb     # Full implementation
├── 📄 README.md                # Documentation
├── 📁 faces_age_detection/     # Saved model files
└── 📁 assets/                  # Plots and visuals

---
#### 3️⃣ Real-Time Emotion Recognition
- **Languages:** Python  
- **Libraries:** 
  - Deep Learning: `keras`, `tensorflow`
  - Computer Vision: `opencv-python`
  - Pre-trained Models: `MobileNet`
  - Data preprocessing: `numpy`, `matplotlib`
- **Techniques:** 
  - Transfer learning using MobileNet
  - Image data augmentation
  - Real-time face detection and emotion recognition
  - Model checkpointing and early stopping
- **Environment:** Google Colab / Local Python

---
