"""
Ultralytics Integration Script
"""
from ultralytics import YOLO


def get_ultralytics_model(name, num_classes=None, pretrained=True):
    """
    Initialize an Ultralytics model for object detection or segmentation.

    Args:
        name (str): Model name (e.g., 'yolov8n' for YOLOv8 nano model).
        num_classes (int, optional): Number of classes for the model.
                                     Default is None .
        pretrained (bool): Whether to load pretrained weights. (Default = True)

    Returns:
        model: An Ultralytics YOLO model instance.
    """
    model = YOLO(name, pretrained=pretrained)
    if num_classes is not None:
        model.model.head[-1].nc = num_classes

    return model
