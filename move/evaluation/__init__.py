"""MoVE Evaluation - Validation and Performance Analysis

This module houses evaluation utilities for comparing MoVE outputs
with baseline Llama activations and measuring performance metrics."""

from .validation import (
    MoVEValidator,
    BaselineExtractor,
    ValidationMetrics,
    create_test_dataset,
    run_validation_example
)

__all__ = [
    'MoVEValidator',
    'BaselineExtractor', 
    'ValidationMetrics',
    'create_test_dataset',
    'run_validation_example'
]