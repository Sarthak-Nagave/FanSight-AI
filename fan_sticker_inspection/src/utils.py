import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_prediction_result(class_name, confidence, threshold):
    if confidence < threshold:
        return "UNCERTAIN"
    
    if class_name == "STICKER_PRESENT":
        return "PASS"
    elif class_name == "STICKER_MISSING":
        return "FAIL"
    
    return "UNCERTAIN"
