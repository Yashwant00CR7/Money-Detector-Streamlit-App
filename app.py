import streamlit as st
import base64
from PIL import Image
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf
import traceback

# Define class names for CNN
cnn_classes = ['10 Rupees', '100 Rupees', '20 Rupees', '200 Rupees', '50 Rupees', '500 Rupees']

# ---------------------------------------------------------------------------
# Helper to unwrap input shapes that may be nested containers
# ---------------------------------------------------------------------------
def unwrap_input_shape(input_shape):
    """Recursively unwrap input_shape until it is a flat tuple of ints/None.
    Handles cases like [(None, 7, 7, 1024)] or (None, 7, 7, (1024,)).
    """
    print(f"DEBUG: unwrapping {input_shape} type={type(input_shape)}")
    current = input_shape
    for _ in range(20):  # safety limit
        # Convert tf.TensorShape to a tuple
        if isinstance(current, tf.TensorShape):
            current = tuple(current.as_list())
            continue
        # Unwrap single‑element containers
        if isinstance(current, (list, tuple)):
            if len(current) == 0:
                break
            if len(current) == 1:
                print(f"DEBUG: unwrapping container of len 1: {current} -> {current[0]}")
                current = current[0]
                continue
            # Check if we already have a valid shape tuple
            def is_number_like(x):
                return x is None or isinstance(x, (int, float)) or "Dimension" in str(type(x))
            if all(is_number_like(x) for x in current):
                print(f"DEBUG: identified valid shape tuple: {current}")
                return tuple(current)
            # Handle nested channel dimension (e.g., last element is a container)
            last = current[-1]
            if isinstance(last, (list, tuple, tf.TensorShape)):
                # Convert TensorShape if needed
                if isinstance(last, tf.TensorShape):
                    last = tuple(last.as_list())
                # If last is a single number like (1024,)
                if len(last) == 1 and is_number_like(last[0]):
                    new_shape = list(current[:-1]) + [last[0]]
                    print(f"DEBUG: fixed nested channel dim: {current} -> {new_shape}")
                    current = tuple(new_shape)
                    continue
        # No more unwrapping possible
        break
    # Final fallback conversion
    if isinstance(current, list):
        current = tuple(current)
    print(f"DEBUG: final unwrapped: {current}")
    return current

# ---------------------------------------------------------------------------
# Robust Conv2D to handle Keras 3 shape quirks
# ---------------------------------------------------------------------------
class RobustConv2D(layers.Conv2D):
    def build(self, input_shape):
        print(f"RobustConv2D.build raw input: {input_shape}")
        input_shape = unwrap_input_shape(input_shape)
        # Final safety check – ensure the channel dimension is not a container
        if isinstance(input_shape, (tuple, list)) and len(input_shape) > 0:
            last = input_shape[-1]
            if isinstance(last, (list, tuple, tf.TensorShape)):
                print(f"WARNING: RobustConv2D input_shape still has nested last dim: {last}. Forcing unwrap.")
                # Extract the innermost number
                if isinstance(last, tf.TensorShape):
                    last = tuple(last.as_list())
                if len(last) >= 1 and isinstance(last[-1], (int, float)):
                    input_shape = list(input_shape)
                    input_shape[-1] = last[-1]
                    input_shape = tuple(input_shape)
                    print(f"RobustConv2D forced fix: {input_shape}")
        print(f"RobustConv2D.build calling super with: {input_shape}")
        super().build(input_shape)
        if hasattr(self, "kernel"):
            print(f"RobustConv2D built kernel with shape: {self.kernel.shape}")

    def call(self, inputs):
        # Unwrap list/tuple wrapper around the tensor (Keras sometimes passes a list)
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 1:
                inputs = inputs[0]
                print("RobustConv2D.call unwrapped list wrapper around input tensor")
            else:
                print(f"WARNING: RobustConv2D.call received unexpected list/tuple of length {len(inputs)}")
        return super().call(inputs)

# ---------------------------------------------------------------------------
# Custom attention layer used in the original model
# ---------------------------------------------------------------------------
class CentralFocusSpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(CentralFocusSpatialAttention, self).__init__(**kwargs)
        self.conv_attention = None
        self.gamma = None

    def build(self, input_shape):
        print(f"CentralFocusSpatialAttention.build raw input: {input_shape}")
        input_shape = unwrap_input_shape(input_shape)
        # Build internal conv layer (expects 2‑channel input)
        self.conv_attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        self.gamma = self.add_weight(name='gamma', shape=(), initializer='zeros', trainable=True)
        if input_shape is not None and len(input_shape) > 0 and input_shape[-1] is not None:
            conv_input_shape = list(input_shape)
            conv_input_shape[-1] = 2
            self.conv_attention.build(tuple(conv_input_shape))
        super().build(input_shape)

    def call(self, inputs):
        # Channel attention
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv_attention(concat)
        # Gaussian mask
        shape = tf.shape(inputs)
        height, width = shape[1], shape[2]
        center_x = height // 2
        center_y = width // 2
        sigma = tf.cast(height / 4, tf.float32)
        x = tf.range(0, height, dtype=tf.float32)
        y = tf.range(0, width, dtype=tf.float32)
        x_mask = tf.exp(-(x - tf.cast(center_x, tf.float32)) ** 2 / (2 * sigma ** 2))
        y_mask = tf.exp(-(y - tf.cast(center_y, tf.float32)) ** 2 / (2 * sigma ** 2))
        gaussian_mask = tf.tensordot(x_mask, y_mask, axes=0)
        gaussian_mask = tf.expand_dims(gaussian_mask, axis=-1)
        gaussian_mask = tf.expand_dims(gaussian_mask, axis=0)
        gaussian_mask = tf.cast(gaussian_mask, dtype=inputs.dtype)
        attention_weighted = attention * gaussian_mask
        return inputs * (1 + self.gamma * attention_weighted)

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
@st.cache_resource
def load_cnn_model():
    from tensorflow.keras.models import load_model as _load_model
    with CustomObjectScope({
        'CentralFocusSpatialAttention': CentralFocusSpatialAttention,
        'Conv2D': RobustConv2D
    }):
        return _load_model('Currency_Detection_model_with_DenseNet121_and_CentralFocusSpatialAttention.h5', compile=False)

@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO as _YOLO
    return _YOLO('runs/detect/train4/weights/best.pt')

INPUT_IMAGE_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("Currency Detection App")
st.write("Upload an image to detect currency.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        img_array = np.array(img)
        with st.spinner("Loading models..."):
            yolo_model = load_yolo_model()
            cnn_model = load_cnn_model()
        yolo_results = yolo_model.predict(source=img_array, save=False)
        boxes = yolo_results[0].boxes
        class_names = yolo_results[0].names
        detected_currency = False
        for box in boxes:
            cls = int(box.cls[0])
            if class_names[cls] == "Currency":
                detected_currency = True
                break
        if detected_currency:
            st.success("Currency detected with YOLO model!")
            img_resized = img.resize(INPUT_IMAGE_SIZE)
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            cnn_prediction = cnn_model.predict(img_array)[0]
            predicted_class_index = np.argmax(cnn_prediction)
            predicted_class_name = cnn_classes[predicted_class_index]
            confidence = cnn_prediction[predicted_class_index] * 100
            st.write(f"Predicted Currency: **{predicted_class_name}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
        else:
            st.error("No currency detected by YOLO model.")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.code(traceback.format_exc())
