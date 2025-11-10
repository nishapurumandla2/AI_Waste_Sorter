# realtime_detect.py
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model/waste_sorter_cnn.h5')

# Class names
classes = ['Organic', 'Recyclable']

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

print("🎥 Webcam started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from webcam.")
        break

    # Flip the frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Define the green bounding box (center)
    box_size = 220
    x1 = width // 2 - box_size // 2
    y1 = height // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Draw the green square
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Extract region of interest (ROI) inside the box
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (128, 128))
    roi_normalized = roi_resized.astype('float32') / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)

    # Predict class
    prediction = model.predict(roi_expanded, verbose=0)
    class_index = int(prediction[0][0] > 0.5)
    confidence = prediction[0][0] * 100 if class_index == 1 else (100 - prediction[0][0] * 100)

    label = classes[class_index]
    color = (0, 255, 0) if class_index == 1 else (0, 165, 255)

    # Display text above the box
    text = f"{label} ({confidence:.1f}%)"
    cv2.putText(frame, text, (x1 - 40, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    # Instructions
    cv2.putText(frame, "Place object inside green box", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show video
    cv2.imshow("AI Waste Sorter - Live", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
