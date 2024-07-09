
import argparse
import io
from PIL import Image
import datetime
import numpy as np
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response, jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import warnings
warnings.filterwarnings('ignore')
import sys
import json
import skimage.draw
import cv2
import random
import math
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils



# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Root directory of the project
ROOT_DIR = "C:\\Users\\anush\\Instance-Segmentation-App-Using-Flask-And-Mask-R-CNN"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"


    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU =1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + Hard_hat, Safety_vest

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.82

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config,model_dir=DEFAULT_LOGS_DIR)


################################################################################

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')


def calculate_area(mask):
    return np.sum(mask)

@app.route("/predict", methods=["POST"])
def predict():
    # Load the image

    ref_area_cm2 = 50
    image_data = request.files["image"].read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    inference_config = InferenceConfig()
    
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=inference_config,model_dir=DEFAULT_LOGS_DIR)
    model_path = 'mask_rcnn_object_0250.h5'
    
    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    class_names = ['BG', 'wound', 'ruler']
    # Run inference
    results = model.detect([image], verbose=1)

    # Visualize the results
    r = results[0]
    masks = r["masks"]
    class_ids = r["class_ids"]
    scores = r["scores"]
    boxes = r['rois']


    high_score_indices = [i for i, score in enumerate(r['scores']) if score > 0.83]
    filtered_rois = r['rois'][high_score_indices]
    filtered_masks = r['masks'][:, :, high_score_indices]
    filtered_class_ids = r['class_ids'][high_score_indices]
    filtered_scores = r['scores'][high_score_indices]

    # Find the ruler in the detected objects
    ruler_index = None
    for i, class_id in enumerate(filtered_class_ids):
        if class_names[class_id] == 'ruler':
            ruler_index = i
            break

    areas = []
    if ruler_index is not None:
        # Calculate area of the ruler
        ruler_mask = filtered_masks[:, :, ruler_index]
        ref_area_pixels = calculate_area(ruler_mask)
        print(f"Area of the ruler: {ref_area_pixels} pixels")

        # Calculate the conversion factor from pixels to cm²
        conversion_factor = ref_area_cm2 / ref_area_pixels

        # Calculate the areas of other objects
        for i in range(filtered_masks.shape[-1]):
            if i != ruler_index:
                mask = filtered_masks[:, :, i]
                area_pixels = calculate_area(mask)
                area_cm2 = area_pixels * conversion_factor
                areas.append(area_cm2)
                print(f"Area of mask {i}: {area_pixels} pixels, {area_cm2:.2f} cm², Score: {filtered_scores[i]}")
    else:
        print("Ruler not found in the image.")

    # Prepare visualization (optional)
    visualize.display_instances(image, filtered_rois, filtered_masks, filtered_class_ids, 
                                class_names, filtered_scores, figsize=(5, 5))
    response_data = {
        'areas': areas,
        'class_ids': filtered_class_ids.tolist(),
        'scores': filtered_scores.tolist()
    }
    return jsonify(response_data)
    # Render the template with the image and areas
    # return render_template('index.html', image_data=image_data, areas=areas)

    
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                # class_names, r['scores'],figsize=(5,5))
                                                          
    # return render_template('index.html')
    #return render_template('index.html', image_path=image_path)
    # Call the display_instances() function to generate the output image
    #fig = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset_train.class_names, r['scores'], figsize=(5,5))
    
    # Define the output path for the image relative to the current working directory
    #output_path = os.path.join(os.getcwd(), 'output_image.png')
    
    # Save the output image to the current working directory
    #fig.savefig(output_path)
    
    # Close the Matplotlib figure to free up memory
    #plt.close(fig)
                                
      

if __name__ == "__main__":
    app.run(debug=True)

    

