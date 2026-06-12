"""Feature extraction backends for visual localization."""

from gps_ex1.features.global_descriptors import (
    BasicDescriptorExtractor,
    DescriptorExtractor,
    DinoV2DescriptorExtractor,
    create_descriptor_extractor,
)

__all__ = [
    "BasicDescriptorExtractor",
    "DescriptorExtractor",
    "DinoV2DescriptorExtractor",
    "create_descriptor_extractor",
]
