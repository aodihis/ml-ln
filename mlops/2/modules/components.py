
"""Initiate tfx pipeline components
"""

import os
from dataclasses import dataclass

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher, Tuner
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

@dataclass
class PipelineConfig:
    """ Configuration object for pipeline components. """
    data_dir: str
    transform_module: str
    training_module: str
    tuner_module: str
    training_steps: int
    eval_steps: int
    serving_model_dir: str

def init_components(
    config: PipelineConfig
):
    """Initiate tfx pipeline components

    Args:
        config (PipelineConfig): a config

    Returns:
        TFX components
        :param config:
        :param tuner_module:
    """


    # Configure the train-test split ratio (80-20)
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )

    example_gen = CsvExampleGen(
        input_base=config.data_dir,
        output_config=output
    )

    statistics_gen = StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"]
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(config.transform_module)
    )

    tuner = Tuner(
        module_file=os.path.abspath(config.tuner_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config.training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=config.eval_steps),
    )

    trainer = Trainer(
        module_file=os.path.abspath(config.training_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config.training_steps),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=config.eval_steps)
    )



    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

    # Define metrics for multi-class classification
    metrics_specs = [
        tfma.MetricsSpec(
            metrics=[
                # Classification metrics
                tfma.MetricConfig(class_name='SparseCategoricalAccuracy',
                                threshold=tfma.MetricThreshold(
                                    value_threshold=tfma.GenericValueThreshold(
                                        lower_bound={'value': 0.6}),
                                    change_threshold=tfma.GenericChangeThreshold(
                                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                        absolute={'value': -0.0001})
                                )),
                # Add mean prediction
                tfma.MetricConfig(class_name='MeanPrediction'),
                # Add example count for monitoring
                tfma.MetricConfig(class_name='ExampleCount')
            ],
        )
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name='serving_default',
                label_key='Quality_Label_xf',
                preprocessing_function_names=['transform_features'],
            )
        ],
        slicing_specs=[
            # Overall metrics
            tfma.SlicingSpec(),
        ],
        metrics_specs=metrics_specs
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config.serving_model_dir
            )
        ),
    )

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )

    return components
