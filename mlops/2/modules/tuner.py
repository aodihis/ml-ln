"""Hyperparameter tuning module."""
from typing import Any, Dict
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.v1.components import TunerFnResult

from modules.transform import NUMERICAL_FEATURES, transformed_name
from modules.trainer import input_fn

def model_builder(hp: kt.HyperParameters) -> tf.keras.Model:
    """Build model with hyperparameter tuning.
    
    Args:
        hp: Hyperparameters to tune
        
    Returns:
        Compiled Keras model
    """
    input_features = [
        tf.keras.Input(shape=(1,), name=transformed_name(key))
        for key in NUMERICAL_FEATURES
    ]
    concatenate = tf.keras.layers.concatenate(input_features)

    # Define architecture with hyperparameters
    layers_config = [
        (hp.Choice('unit_1', [128, 256]), hp.Choice('dropout_1', [0.2, 0.4])),
        (hp.Choice('unit_2', [64, 128]), hp.Choice('dropout_2', [0.2, 0.4])),
        (hp.Choice('unit_3', [32, 64]), hp.Choice('dropout_3', [0.2, 0.4]))
    ]

    deep = concatenate
    for units, dropout_rate in layers_config:
        deep = tf.keras.layers.Dense(units, activation="relu")(deep)
        deep = tf.keras.layers.Dropout(dropout_rate)(deep)

    outputs = tf.keras.layers.Dense(3, activation="softmax")(deep)
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [0.0001, 0.001])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model

def tuner_fn(fn_args: Dict[str, Any]) -> TunerFnResult:
    """Build tuner using the KerasTuner API.
    
    Args:
        fn_args: Hyperparameter tuning arguments
        
    Returns:
        Tuner results including best hyperparameters
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 32)

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_sparse_categorical_accuracy',
        max_trials=5,
        directory=fn_args.working_dir,
        project_name='cinnamon_classification'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'epochs': 10
        }
    )
