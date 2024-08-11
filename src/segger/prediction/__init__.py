"""
Prediction module for Segger.

Contains prediction scripts and utilities for the Segger model.
"""

__all__ = [
    "LitSegger", 
    "load_model", 
    "predict"
    ]

from .predict import LitSegger, load_model, predict
