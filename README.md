# 🍕 Food Recognition & Calorie Estimation (Task-05)

An automated Computer Vision and Machine Learning application built using **Python**, **OpenCV**, and **Scikit-Learn** that recognizes food items from input images (*Pizza, Hamburger, Samosa*) and estimates their calorie content per serving.

---

## 📌 Project Overview
This project processes image samples of popular food items, extracts flat pixel feature vectors, and uses a classification algorithm to identify the food type. Once recognized, it maps the class label to a calorie lookup table to present an estimated calorie count to the user.

---

## 📸 Sample Dataset Categories & Calorie Mapping

The model is configured to recognize the following sample food items and estimate their standard portion energy values:

| Food Category | Standard Portion | Estimated Calories |
| :--- | :--- | :--- |
| 🍕 **Pizza** | 1 Slice | **~266 kcal** |
| 🍔 **Hamburger** | 1 Burger | **~295 kcal** |
| 🥟 **Samosa** | 1 Piece | **~262 kcal** |

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.14+
* **Computer Vision:** OpenCV (`cv2`)
* **Machine Learning:** Scikit-Learn (`KNeighborsClassifier`)
* **Data Processing & Plotting:** NumPy, Matplotlib

---

## ⚙️ Repository Structure
```text
ML_Task5/
│── custom_food_data/           # Folder containing pizza, hamburger, samosa images
│── food_calorie_estimator.py   # Main classification & calorie estimation script
│── .gitignore                  # Prevents raw image datasets from uploading
└── README.md                   # Project documentation
