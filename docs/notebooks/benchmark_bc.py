import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import numpy as np
from typing import Dict
from segger.validation.utils import *

# Define colors for methods
method_colors = {
    'segger': '#D55E00',
    'segger_n0': '#E69F00',
    'segger_n1': '#F0E442',
    'Baysor': '#0072B2',
    '10X': '#009E73',
    '10X-nucleus': '#CC79A7',
    'BIDCell': '#8B008B'
}

# Hard-coded initialization example
segmentation_paths = {
    'segger': benchmarks_path / 'adata_segger.h5ad',
    'Baysor': benchmarks_path / 'adata_baysor.h5ad',
    '10X': benchmarks_path / 'adata_10X.h5ad',
    '10X-nucleus': benchmarks_path / 'adata_10X_nuc.h5ad',
    'BIDCell': benchmarks_path / 'adata_BIDCell.h5ad'
}

# Load the segmentations
segmentations_dict = load_segmentations(segmentation_paths)
# Generate individual plots and the summary plot
plot_general_statistics_plots(segmentations_dict, output_path)
