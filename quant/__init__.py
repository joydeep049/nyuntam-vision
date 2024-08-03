def initialize_initialization(algoname, task):
    if algoname == "FXQuant":
        from .torch import FXQuant

        return FXQuant
    elif algoname == "NNCF":
        if task == "image_classification":
            from .nncf import NNCFClassifcation

            return NNCFClassifcation
        elif task == "object_detection":
            from .nncf import NNCFObjectDetection

            return NNCFObjectDetection

    elif algoname == "TensorRT":
        from .tensorrt import TensorRT

        return TensorRT
    elif algoname == "ONNXQuant":
        from .onnx import ONNXQuant, DummyDataReader

        return ONNXQuant
    elif algoname == "NNCFQAT":
        if task == "image_classification":
            from .nncf import NNCFQATClassification

            return NNCFQATClassification
        elif task == "object_detection":
            from .nncf import NNCFQATObjectDetection

            return NNCFQATObjectDetection
    elif algoname == "TensorRTQAT":
        from .tensorrt import TensorRTQAT
        return TensorRTQAT

    else:
        return None
