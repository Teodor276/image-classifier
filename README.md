# Face Expression Classifier – Machine Learning Deployment Project

## 📖 Project Overview

This project covers **end-to-end ML deployment**:

- Building, tuning, and evaluating multiple models on a facial expression dataset.
- Selecting the best-performing model based on test metrics.
- Retraining the best model on the **entire dataset** for optimal performance.
- Exporting the trained model using `pickle`.
- Deploying the model via **FastAPI** so it can accept images and return predictions.

---

## 🚀 Live Demo

👉 [Click here to try the live demo](https://image-classifier-hb1g.onrender.com/)

> ⏳ **Please wait 30–60 seconds on first load.**  
> The app is hosted on Render’s free tier, so it spins down when inactive and needs time to restart.

---


## 🛠️ What This Repository Contains

### 1️⃣ Data Cleaning

- Loads the dataset, checks for missing values, and ensures consistent formatting.
- Performs **feature engineering** where needed.
- Saves the cleaned data for model training.

### 2️⃣ Model Building

- Trains multiple ML models, including Random Forest and SVM.
- Performs hyperparameter tuning using GridSearchCV.
- Compares models using different metrics.
- Selects the **best model**, then retrains it on 100% of the dataset for final export.

### 3️⃣ Backend Deployment

- Uses **FastAPI** to build a REST API that accepts uploaded images.
- The backend performs image preprocessing (grayscale, face detection, wavelet transform).
- The preprocessed image is fed into the trained model to return predicted labels and probabilities.
  
---

## 🖼️ Image Handling with OpenCV

This project uses the **`cv2` (OpenCV)** library for:

- **Face and eye detection:** Utilizes OpenCV’s pre-trained Haar cascade classifiers to detect faces and eyes within an image.
- **Preprocessing:** Converts images to grayscale and applies wavelet transforms to enhance feature extraction before passing the image to the model.

⚠️ **Limitation:** The model requires a face with **exactly two eyes detected** to make a prediction. If a face or both eyes are not detected, the image is discarded. This is a limitation of OpenCV’s Haar cascade detection, which may fail on images with poor lighting, occlusions, or extreme angles.

---

## ✅ Conclusion

This project demonstrates a **complete machine learning deployment pipeline**, starting from data cleaning and model training to real-world deployment using FastAPI. By incorporating **OpenCV for image preprocessing**, we enable the application to handle user-submitted images effectively, ensuring meaningful predictions only when valid face and eye detections are made.


---

## 🎓 Course Context

This project is part of a **Machine Learning course** offered by **CodeBasics**. The course provides comprehensive knowledge on the fundamentals of machine learning, with hands-on experience in building real-world applications. This project helped solidify core concepts such as model development, data preprocessing, and deployment, offering practical insights into the entire machine learning pipeline.

---


