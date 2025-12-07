"""
NFL Prediction Engine - Colab Module

Modular scripts for training, prediction, and analysis in Google Colab.

Usage:
    from colab import setup, data, train, predict, betting, save
"""

from . import setup
from . import data
from . import train
from . import predict
from . import betting
from . import save

__all__ = ['setup', 'data', 'train', 'predict', 'betting', 'save']
