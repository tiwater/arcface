# arcFace.py

from rknnlite.api import RKNNLite
from PIL import Image
import numpy as np
import os

# Initialize RKNN model
rknn = RKNNLite()
rknn.load_rknn('./model/arcface.rknn')
rknn.init_runtime()

# Preprocess the source image
img_src = Image.open('./dataset/testImg/0204.jpg')
img_src = img_src.resize((112, 112))
img_src = img_src.convert('RGB')

img_src = np.array(img_src).astype(np.float32)
img_src = img_src[np.newaxis, ...]

# Get inference for the source image
output_src = rknn.inference(inputs=[img_src], data_format='nhwc')[0][0]

# Directory containing target images
target_dir = './dataset/testImg'

# Iterate over all .jpg files in the target directory
for filename in os.listdir(target_dir):
    
    if filename.endswith('.jpg'):
        # Preprocess each target image
        img_target = Image.open(os.path.join(target_dir, filename))
        img_target = img_target.resize((112, 112))
        img_target = img_target.convert('RGB')
        
        img_target = np.array(img_target).astype(np.float32)
        img_target = img_target[np.newaxis, ...]

        # Get inference for the target image
        output_target = rknn.inference(inputs=[img_target],
                                       # Format of the input data
                                       data_format='nhwc')[0][0]

        # Calculate the cosine similarity
        cos_sim = np.dot(output_src, output_target) / (np.linalg.norm(output_src) * np.linalg.norm(output_target))

        # # Calculate the L2 distance (Mean Squared Error)
        mse_dist = np.linalg.norm(output_src - output_target)

        # Print the results
        print(f"File: {filename}")
        print(f"Cosine Sim: {cos_sim}")
        print(f"MSE Distance: {mse_dist}\n")