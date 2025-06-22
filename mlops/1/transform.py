import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "Personality"

# Feature keys
NUMERIC_FEATURE_KEYS = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]

BINARY_CATEGORICAL_FEATURE_KEYS = [
    "Stage_fear",
    "Drained_after_socializing"
]

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Returns:
        outputs: map from feature keys to transformed features.
    """
    outputs = {}

    # Normalize numeric features
    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])

    # Encode binary categorical features: Yes -> 1, No -> 0
    for key in BINARY_CATEGORICAL_FEATURE_KEYS:
        outputs[transformed_name(key)] = tf.cast(tf.equal(inputs[key], "Yes"), tf.int64)

    # Encode label: Extrovert = 1, Introvert = 0
    outputs[transformed_name(LABEL_KEY)] = tf.cast(tf.equal(inputs[LABEL_KEY], "Extrovert"), tf.int64)

    return outputs
