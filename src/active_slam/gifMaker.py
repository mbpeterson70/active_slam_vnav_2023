
import os
from PIL import Image

def create_gif(image_folder, output_file):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            images.append(image)

    images[0].save(output_file, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)

# Usage example
image_folder = "/home/annika/data/airsim_output/blob_tracking_experiment_3_figures/airsim_100_200"
output_file = "/home/annika/data/airsim_output/blob_tracking_experiment_3.gif"
create_gif(image_folder, output_file)
