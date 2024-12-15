"""
Prediction module for Segger.

Contains prediction scripts and utilities for the Segger model.
"""

__all__ = ["load_model", "segment"]

from .predict_parquet import load_model, segment
