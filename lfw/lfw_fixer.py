import os
import shutil

in_path = "..\\..\\Datasets\\LFW3D\\LFW3D.0.1.1"
out_path = "lfw_data"

for subdir in os.listdir(in_path):
    for filename in os.listdir(os.path.join(in_path, subdir)):
        shutil.copy(os.path.join(in_path, subdir, filename), os.path.join(out_path, filename))

