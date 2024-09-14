# Refer to https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu
import onnx
from onnx import shape_inference

# Load your ONNX model
model_path = "./model/arcface.onnx"
model = onnx.load(model_path)

# Perform shape inference
inferred_model = shape_inference.infer_shapes(model)

# Save the inferred model to a file
inferred_model_path = "./model/arcface_inferred.onnx"
onnx.save(inferred_model, inferred_model_path)