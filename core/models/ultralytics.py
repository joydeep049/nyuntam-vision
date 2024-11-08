"""
Ultralytics Integration Script
"""
from ultralytics import YOLO


def get_ultralytics_model(
    name, num_classes=None, custom_model_path=None):
    """
    Initialize an Ultralytics model with options for custom weights or stock pretrained weights.

    Args:
        name (str): Model name (e.g., 'yolov8n.pt' for YOLOv8 nano model).
        num_classes (int, optional): Number of classes for the model. Default is None (use default from Ultralytics).
        custom_model_path (str, optional): Path to custom model weights. If provided, loads this model instead.
        pretrained (bool): Whether to load stock pretrained weights (used only if custom_model_path is None).

    Returns:
        model: An Ultralytics YOLO model instance.
    """
    if custom_model_path:
        model = YOLO(custom_model_path)  # Custom Weight Loading
    else:
        model = YOLO(name)

    if num_classes is not None:
        model.model.head[-1].nc = num_classes

    return model
