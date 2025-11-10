# ♻️ AI Waste Sorter  

## 🔍 Overview  
The **AI Waste Sorter** is a machine learning project designed to automatically classify waste into **Organic** and **Recyclable** categories using **Computer Vision and Deep Learning (CNN)**.  
The system helps improve waste management efficiency by enabling real-time detection through a webcam or image input — promoting a sustainable and eco-friendly environment.

---

## 👩‍💻 Developer  
**Name:** Nisha Purumandla  
**Roll Number:** 23BD1A05DD  
**Institution:** [Add your college name if required]  
**Project Duration:** 3 Weeks  

---

## 🧠 Abstract  
Improper waste segregation is a major global challenge that affects recycling and contributes to pollution.  
This project presents an **AI-powered waste classification system** that uses a **Convolutional Neural Network (CNN)** model trained on labeled waste images.  
It can detect and classify waste in real-time using a camera feed, identifying whether the item belongs to the *Organic* or *Recyclable* category.  

The model can also be integrated with smart waste bins or IoT devices for automated sorting, helping cities move closer to sustainable smart-city goals.

---

## 🚀 Features  
✅ Classifies waste as **Organic** or **Recyclable**  
✅ Supports **real-time webcam detection**  
✅ Uses **TensorFlow/Keras CNN** and **MobileNetV2 (Transfer Learning)**  
✅ Works with custom image datasets  
✅ Lightweight and easy to deploy  
✅ Scalable for smart city waste systems  

---

## 🧩 Tech Stack  
| Category | Technology |
|-----------|-------------|
| **Programming Language** | Python |
| **Libraries Used** | TensorFlow, Keras, OpenCV, NumPy, Matplotlib |
| **Model Used** | CNN / MobileNetV2 (Transfer Learning) |
| **Tools** | VS Code, PowerShell, GitHub |
| **Dataset** | Custom/Kaggle Waste Classification Dataset |
| **Hardware** | Laptop webcam for live detection |

---

## ⚙️ Installation and Setup  

### 1️⃣ Clone this repository
```bash
git clone https://github.com/nishapurumandla2/AI_Waste_Sorter.git
cd AI_Waste_Sorter
2️⃣ Create and activate a virtual environment (optional)
python -m venv .venv
.venv\Scripts\activate   # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model (optional — already included)
python main.py

5️⃣ Run image prediction
python predict.py

6️⃣ Run real-time detection
python realtime_detect.py

🧱 Project Structure
AI_Waste_Sorter/
│
├── dataset/
│   ├── Organic/
│   └── Recyclable/
│
├── model/
│   └── waste_sorter_cnn.h5
│
├── main.py                # Model training
├── predict.py             # Predict using saved model
├── realtime_detect.py     # Real-time webcam detection
├── requirements.txt
└── README.md
```
🧬 How It Works

The system captures input images from the webcam or dataset.
Images are preprocessed and passed to the trained CNN/MobileNetV2 model.
The model classifies the image as Organic or Recyclable.
For real-time mode, predictions are displayed live with labels and confidence scores.

🌱 Future Enhancements

Integration with smart waste bins using IoT
Adding more waste categories (plastic, glass, metal, etc.)
Deploying as a mobile or web app
Using Edge AI for offline waste detection

🏁 Conclusion

The AI Waste Sorter project demonstrates the potential of Artificial Intelligence to enhance waste management systems through automation and real-time decision-making.
It promotes sustainability and provides a foundation for future smart waste segregation systems.


🏷️ License

This project is developed as part of an academic submission and is open for educational use.


---

