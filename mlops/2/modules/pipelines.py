"""Pipeline configuration module."""
from typing import Any, List, Text
from absl import logging
from tfx.orchestration import metadata, pipeline

def init_pipeline(
    components: List[Any],
    pipeline_root: Text,
    pipeline_name: Text,
    metadata_path: Text
) -> pipeline.Pipeline:
    """Initialize TFX pipeline.
    
    Args:
        components: List of TFX components to include in the pipeline
        pipeline_root: Root directory for pipeline artifacts
        pipeline_name: Name of the pipeline
        metadata_path: Path to the metadata DB
    
    Returns:
        A TFX pipeline object
    """
    logging.info("Pipeline root set to: %s", pipeline_root)
    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args
    )
