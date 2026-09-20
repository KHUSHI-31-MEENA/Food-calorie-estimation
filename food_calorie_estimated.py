import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------
# Step 1: Calorie Lookup Dictionary
# ---------------------------------------------------------
CALORIE_DICT = {
    'pizza': 266,
    'hamburger': 295,
    'samosa': 262
}

# Change this path according to your PC
DATA_DIR = r"C:\Users\yash7\Downloads\PRODIGY_ML_05-main\PRODIGY_ML_05-main\custom_food_data"

IMG_SIZE = 64

X = []
y = []
labels_dict = {}
label_count = 0

print("Loading images...")

# Check dataset folder
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset folder not found:\n{DATA_DIR}")

# Read images
for food in os.listdir(DATA_DIR):

    food_path = os.path.join(DATA_DIR, food)

    if not os.path.isdir(food_path):
        continue

    labels_dict[label_count] = food.lower()

    image_count = 0

    for img_name in os.listdir(food_path):

        img_path = os.path.join(food_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read: {img_path}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img.flatten())
        y.append(label_count)

        image_count += 1

    print(f"{food} -> {image_count} images loaded")

    label_count += 1

X = np.array(X)
y = np.array(y)

print("\nTotal Images:", len(X))

# Stop if dataset empty
if len(X) == 0:
    raise ValueError(
        "No images found!\nCheck DATA_DIR path and folder structure."
    )

# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------

model = KNeighborsClassifier(n_neighbors=1)

model.fit(X, y)

print("Model trained successfully!")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

sample_idx = np.random.randint(len(X))

sample_img = X[sample_idx]

prediction = model.predict(sample_img.reshape(1, -1))[0]

food_name = labels_dict[prediction]

calories = CALORIE_DICT.get(food_name, "Unknown")

img = sample_img.reshape(IMG_SIZE, IMG_SIZE, 3)
img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title(f"{food_name.upper()}\nEstimated Calories: {calories} kcal")
plt.axis("off")
plt.show()