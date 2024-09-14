# Refer to https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu
from PIL import Image
import numpy as np

import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from pathlib import Path

def preprocess_image(image_path):
    # Open image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    # Resize image to the desired size (e.g., 112x112)
    img = img.resize((112, 112))
    # Convert image to numpy array
    img_array = np.array(img).astype(np.float32)
    # Normalize to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    # Change data layout: from (height, width, channels) to (channels, height, width)
    img_array = np.transpose(img_array, (2, 0, 1))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

class ImageDataReader(CalibrationDataReader):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_paths = list(Path(image_folder).glob("*.jpg"))
        self.image_index = 0
        self.preprocess_fn = preprocess_image
        self.input_name = None  # will be set later

    def get_next(self):
        if self.image_index < len(self.image_paths):
            image_path = self.image_paths[self.image_index]
            self.image_index += 1
            data = self.preprocess_fn(image_path)
            return {self.input_name: data}
        else:
            return None

    def get_input_name(self, model_path):
        # Load the ONNX model
        model = onnx.load(model_path)
        # Get the name of the first input tensor
        self.input_name = model.graph.input[0].name

# Define paths
model_path = "./model/arcface_inferred.onnx"
quantized_model_path = "./model/arcface_int8.onnx"
image_folder = "./dataset/faces_ms1m_lfw"  # Folder containing JPG images for calibration

# Initialize the data reader
data_reader = ImageDataReader(image_folder)
data_reader.get_input_name(model_path)

# Quantize the model
quantize_static(
    model_input=model_path,
    model_output=quantized_model_path,
    calibration_data_reader=data_reader,
    quant_format=QuantType.QInt8,
    per_channel=True,
    weight_type=QuantType.QInt8
)
print(f"Quantized model saved to: {quantized_model_path}")