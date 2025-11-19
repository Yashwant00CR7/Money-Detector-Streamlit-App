# Currency Detection App

A Streamlit-based web application for detecting and classifying Indian currency notes using deep learning models.

## Features

- **YOLO Object Detection**: Detects currency notes in uploaded images
- **CNN Classification**: Classifies detected currency into denominations (10, 20, 50, 100, 200, 500 Rupees)
- **Custom Attention Mechanism**: Uses CentralFocusSpatialAttention for improved accuracy
- **Interactive Web Interface**: Built with Streamlit for easy image upload and real-time predictions

## Model Architecture

- **Object Detection**: YOLOv8 for currency note detection
- **Classification**: DenseNet121 backbone with custom CentralFocusSpatialAttention layer
- **Input Size**: 224x224 RGB images
- **Output Classes**: 6 Indian currency denominations

## Installation

### Prerequisites
- Python 3.10+
- Conda (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Streamlit_CurrencyDetector
   ```

2. **Create conda environment**
   ```bash
   conda create -n currency_detector python=3.10
   conda activate currency_detector
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload an image** of Indian currency and get instant predictions

## Project Structure

```
Streamlit_CurrencyDetector/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Currency_Detection_model_...h5  # Pre-trained CNN model
├── runs/detect/train4/weights/     # YOLO model weights
│   └── best.pt
└── README.md                       # This file
```

## Requirements

Key dependencies (see `requirements.txt` for full list):
- `streamlit==1.44.1`
- `tensorflow==2.17.0`
- `keras==3.7.0`
- `ultralytics==8.3.108`
- `numpy==1.26.4`
- `Pillow==11.2.1`

**Note**: Keras 3.7.0 is specifically required for compatibility with the saved model.

## Technical Details

### Custom Layers
- **RobustConv2D**: Enhanced Conv2D layer with shape handling for Keras 3 compatibility
- **CentralFocusSpatialAttention**: Custom attention mechanism focusing on central regions with Gaussian weighting

### Model Loading
- Uses `compile=False` for inference-only loading
- Custom object scope for deserializing custom layers

## Troubleshooting

### Common Issues

1. **Shape-related errors**: Ensure you're using Keras 3.7.0 (not 3.12.0+)
2. **Model loading errors**: Clear Streamlit cache with `streamlit cache clear`
3. **YOLO errors**: Verify `runs/detect/train4/weights/best.pt` exists

## License

[Add your license here]

## Acknowledgments

- DenseNet121 architecture from Keras Applications
- YOLOv8 from Ultralytics
- Streamlit for the web framework
