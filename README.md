# Currency Detection App

A Streamlit web app for detecting and classifying Indian currency notes using YOLOv8 + DenseNet121 deep learning models.

**[Try it live →](https://indian-currency-detector.streamlit.app/)**

---

## 🎯 What it does

Upload a photo of an Indian currency note and the app:
1. Detects the note using YOLOv8 object detection
2. Classifies the denomination using a DenseNet121 CNN with custom attention
3. Returns the result instantly in the browser

Supports: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500

---

## 🧠 Model Architecture

- **Detection**: YOLOv8 — trained on 1,000+ custom images, 93.5% mAP@0.5
- **Classification**: DenseNet121 + `CentralFocusSpatialAttention` — custom Gaussian-weighted attention layer
- **Input**: 224×224 RGB images
- **Dataset**: [Indian Currency Dataset on Kaggle](https://www.kaggle.com/datasets/yashwantk23cse/indian-currency)

---

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Conda (recommended)

```bash
git clone https://github.com/Yashwant00CR7/Money-Detector-Streamlit-App.git
cd Money-Detector-Streamlit-App

conda create -n currency_detector python=3.10
conda activate currency_detector

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

---

## 📁 Project Structure

```
├── app.py                          # Streamlit app
├── requirements.txt
├── Currency_Detection_model_...h5  # CNN model weights
└── runs/detect/train4/weights/
    └── best.pt                     # YOLO weights
```

---

## ⚠️ Compatibility Notes

- Requires **Keras 3.7.0** exactly — newer versions break the saved model format
- If you hit shape errors, run `streamlit cache clear` and retry
- Ensure `runs/detect/train4/weights/best.pt` exists before running

---

## 🔗 Related

- Full system with audio feedback: [CurrencyDetector](https://github.com/Yashwant00CR7/CurrencyDetector)
- Kaggle dataset: [Indian Currency](https://www.kaggle.com/datasets/yashwantk23cse/indian-currency)

---

## 📝 License

MIT License
