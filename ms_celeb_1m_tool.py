import os
import numpy as np
import cv2
import pickle

def load_bin(path, image_size):
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_list = []
    for flip in [0, 1]:
        data = np.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]), dtype=np.float32)
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = cv2.imdecode(np.frombuffer(_bin, np.uint8), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = np.flip(img, axis=2)
            data_list[flip][i][:] = img
    data = np.concatenate(data_list)
    return data.astype('float32'), issame_list

# Load images from lfw.bin
image_size = (112, 112)
data, _ = load_bin('./dataset/faces_ms1m_112x112/lfw.bin', image_size)

# Directory to save the images
save_dir = './dataset/lfw'

# Create directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save images to jpg format
num_images = data.shape[0]
images_per_folder = 1000

for i in range(num_images):
    folder_idx = i // images_per_folder
    folder_path = os.path.join(save_dir, f'{folder_idx}')
    
    # Create subdirectory if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    img = data[i].transpose(1, 2, 0).astype(np.uint8)
    filename = os.path.join(folder_path, f'{i % images_per_folder:04d}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)
