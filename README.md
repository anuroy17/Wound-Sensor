# Wound Sensor

## Introduction
Wound Sensor is a deep learning-based application designed to accurately identify and measure third-degree burn wounds. Using Computer Vision techniques, the model provides pixel-level instance segmentation to assist medical professionals in wound assessment and treatment planning.

## Features
- **Deep Learning Model**: Trained to detect and measure surface area of third-degree burn wounds.
- **Instance Segmentation**: Uses Mask R-CNN for precise wound boundary detection.
- **Computer Vision Techniques**: Utilizes OpenCV for image preprocessing and enhancement.
- **Image Pre- processing**: Used VGG Image Annotator for creating test and train dataset.
- **Flask App**: Created Flask web- application for evaluating model performance.


## Tech Stack
- **Primary Programming Language**: Python
- **Frameworks & Libraries**:
  - TensorFlow 2
  - Mask_RCNN
  - OpenCV

## References
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [TensorFlow 2 Object Detection API Documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/)
- [OpenCV Documentation](https://opencv.org/)
- [Mask_RCNN GitHub Repository](https://github.com/matterport/Mask_RCNN)

