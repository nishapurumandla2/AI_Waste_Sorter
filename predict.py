# predict.py
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Load your trained CNN model
model = load_model('model/waste_sorter_cnn.h5')

print("\n♻️  AI Waste Sorter - Image Prediction Mode ♻️")
print("-------------------------------------------------")

# Ask user for image path
img_name = input("Enter image name (e.g., O_5.jpg or R_3.jpg): ").strip()

# Check both folders automatically
possible_paths = [
    os.path.join("dataset", "Organic", img_name),
    os.path.join("dataset", "Recyclable", img_name)
]

img_path = None
for path in possible_paths:
    if os.path.exists(path):
        img_path = path
        break

if img_path is None:
    print(f"❌ Error: Image '{img_name}' not found in dataset folders!")
    exit()

# Load and preprocess image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Make prediction
prediction = model.predict(img_array)
class_index = int(prediction[0][0] > 0.5)

# Output
if class_index == 1:
    label = "♻️ Recyclable Waste"
else:
    label = "🍂 Organic Waste"

confidence = prediction[0][0] * 100 if class_index == 1 else (100 - prediction[0][0] * 100)

print("\n✅ Prediction Results")
print("----------------------")
print(f"📂 Image tested: {img_name}")
print(f"🏷️ Category: {label}")
print(f"📊 Confidence: {confidence:.2f}%\n")
