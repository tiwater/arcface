# arcFace.py

from rknnlite.api import RKNNLite
import onnxruntime as ort
from PIL import Image
import numpy as np
import os


model = ort.InferenceSession('./model/arcface.onnx')

img = Image.open('./faces/src.jpeg')
img = img.resize((112, 112))
img = img.convert('RGB')
img = np.array(img).astype(np.float32)/255.0
img = (img * 2) - 1
img = img.transpose((2, 0, 1))[np.newaxis, ...]

output_src = model.run(None, {'data': img})[0][0]

# Directory containing target images
target_dir = './faces/targets'

# Iterate over all .jpg files in the target directory
for filename in os.listdir(target_dir):
    if filename.endswith('.jpeg'):
        # Preprocess each target image
        img_target = Image.open(os.path.join(target_dir, filename))
        img_target = img_target.resize((112, 112))
        img_target = img_target.convert('RGB')
        img_target = np.array(img_target).astype(np.float32) / 255.0
        img_target = (img_target * 2) - 1
        img_target = img_target.transpose((2, 0, 1))[np.newaxis, ...]

        # Get inference for the target image
        output_target = model.run(None, {'data': img_target})[0][0]

        # Calculate the cosine similarity
        cos_sim = np.dot(output_src, output_target) / (np.linalg.norm(output_src) * np.linalg.norm(output_target))

        # Calculate the L2 distance (Mean Squared Error)
        mse_dist = np.linalg.norm(output_src - output_target) / output_src.shape[0]

        # Print the results
        print(f"File: {filename}")
        print(f"Cosine Sim: {cos_sim}")
        print(f"MSE Distance: {mse_dist}\n")