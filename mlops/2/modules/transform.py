"""Transform module for feature engineering."""
from typing import Dict, List, Text, Any
import tensorflow_transform as tft

NUMERICAL_FEATURES: List[Text] = [
    "Moisture",
    "Ash",
    "Volatile_Oil",
    "Acid_Insoluble_Ash",
    "Chromium",
    "Coumarin"
]

LABEL_KEY: Text = "Quality_Label"
UNUSED_FEATURE: Text = "Sample_ID"
QUALITY_INDEX: Dict[Text, int] = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
}

def transformed_name(key: Text) -> Text:
    """Generate the name for a transformed feature.
    
    Args:
        key: Base feature name
        
    Returns:
        Transformed feature name
    """
    return f"{key}_xf"

def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
    """Preprocess input features into transformed features.
    
    Args:
        inputs: Map from feature keys to raw features
        
    Returns:
        Map from feature keys to transformed features
    """
    outputs: Dict[Text, Any] = {}

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(inputs[LABEL_KEY])

    return outputs
