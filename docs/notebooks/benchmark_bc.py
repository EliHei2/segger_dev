import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import numpy as np
from typing import Dict
from segger.validation.utils import *


benchmarks_path = Path('/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/xe_rep1_bc')
output_path     = benchmarks_path / 'results'
figures_path    = output_path / 'figures'
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


major_colors = {
        'B-cells' : '#d8f55e',
        'CAFs' : '#532C8A',
        'Cancer Epithelial' : '#C72228',
        'Endothelial' : '#9e6762',
        'Myeloid' : '#ffe012',
        'T-cells' : '#3cb44b',
        'Normal Epithelial' : '#0F4A9C',
        'PVL': '#c09d9a',
        'Plasmablasts' : '#000075'
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
segmentations_dict['segger'] = sc.read(benchmarks_path / 'adata_segger.h5ad')
egmentations_dict = {k: segmentations_dict[k] for k in method_colors.keys() if k in segmentations_dict}
scRNAseq_adata = sc.read(benchmarks_path / 'scRNAseq.h5ad')

# Generate individual plots and the summary plot
plot_general_statistics_plots(segmentations_dict, output_path)


markers = find_markers(scRNAseq_adata, cell_type_column='celltype_major', pos_percentile=30, neg_percentile=5)

for method in segmentations_dict.keys():
    segmentations_dict[method] = annotate_query_with_reference(
        reference_adata=scRNAseq_adata,
        query_adata=segmentations_dict[method],
        transfer_column='celltype_major'
    )
    
exclusive_gene_pairs = find_mutually_exclusive_genes(
    adata=scRNAseq_adata,
    markers=markers,
    cell_type_column='celltype_major'
)

mecr_results = {}
for method in segmentations_dict.keys():
    mecr_results[method] = compute_MECR(segmentations_dict[method], exclusive_gene_pairs)
    
    
    
# Step 5: Compute quantized MECR area and counts using the mutually exclusive gene pairs
quantized_mecr_area = {}
quantized_mecr_counts = {}

for method in segmentations_dict.keys():
    if 'cell_area' in segmentations_dict[method].obs.columns:
        quantized_mecr_area[method] = compute_quantized_mecr_area(
            adata=segmentations_dict[method],
            gene_pairs=exclusive_gene_pairs
        )
    quantized_mecr_counts[method] = compute_quantized_mecr_counts(
        adata=segmentations_dict[method],
        gene_pairs=exclusive_gene_pairs
    )
    
plot_mecr_results(mecr_results, output_path=figures_path, palette=method_colors)
plot_quantized_mecr_area(quantized_mecr_area, output_path=figures_path, palette=method_colors)
plot_quantized_mecr_counts(quantized_mecr_counts, output_path=figures_path, palette=method_colors)


new_segmentations_dict = {k: v for k, v in segmentations_dict.items() if k in ['segger', 'Baysor', '10X', '10X-nucleus', 'BIDCell']}

contamination_results = {}

# Loop through the filtered_segmentations_dict to check if 'cell_centroid_x' and 'cell_centroid_y' are present
for method, adata in new_segmentations_dict.items():
    if 'cell_centroid_x' in adata.obs.columns and 'cell_centroid_y' in adata.obs.columns:
        contamination_results[method] = calculate_contamination(
            adata=adata,
            markers=markers,  # Assuming you have a dictionary of markers for cell types
            radius=15,
            n_neighs=20,
            celltype_column='celltype_major',
            num_cells=10000
        )
        
        
boxplot_data = []

for method, df in contamination_results.items():
    melted_df = df.reset_index().melt(id_vars=['Source Cell Type'], var_name='Target Cell Type', value_name='Contamination')
    melted_df['Segmentation Method'] = method
    boxplot_data.append(melted_df)

# Concatenate all the dataframes into one
boxplot_data = pd.concat(boxplot_data)

plot_contamination_results(contamination_results, output_path=figures_path, palette=method_colors)
plot_contamination_boxplots(boxplot_data, output_path=figures_path, palette=method_colors)



segmentations_dict['segger_n1'] = segmentations_dict['segger'][segmentations_dict['segger'].obs.has_nucleus]
segmentations_dict['segger_n0'] = segmentations_dict['segger'][~segmentations_dict['segger'].obs.has_nucleus]
clustering_scores = {}

# Compute clustering scores for all relevant methods
for method, adata in segmentations_dict.items():
    ch_score, sh_score = compute_clustering_scores(adata, cell_type_column='celltype_major')
    clustering_scores[method] = (ch_score, sh_score)



plot_umaps_with_scores(segmentations_dict, clustering_scores, figures_path, palette=major_colors)



for method, adata in segmentations_dict.items():
    if 'spatial' in list(adata.obsm.keys()):
        compute_neighborhood_metrics(adata, radius=15, n_neighs=50, celltype_column='celltype_major')
    
entropy_boxplot_data = []

for method, adata in segmentations_dict.items():
    if 'neighborhood_entropy' in adata.obs.columns:  # Corrected typo here
        entropy_df = pd.DataFrame({
            'Cell Type': adata.obs['celltype_major'],
            'Neighborhood Entropy': adata.obs['neighborhood_entropy'],
            'Segmentation Method': method
        })
        # Filter out NaN values, keeping only the subsetted cells
        entropy_df = entropy_df.dropna(subset=['Neighborhood Entropy'])
        entropy_boxplot_data.append(entropy_df)

# Concatenate all the dataframes into one
entropy_boxplot_data = pd.concat(entropy_boxplot_data)

plot_entropy_boxplots(entropy_boxplot_data, figures_path, palette=method_colors)



purified_markers = find_markers(scRNAseq_adata, 'celltype_major', pos_percentile=20, percentage=75)

sensitivity_results_per_method = {}

# Calculate sensitivity for each segmentation method
for method, adata in segmentations_dict.items():
    sensitivity_results = calculate_sensitivity(adata, purified_markers, max_cells_per_type=2000)
    sensitivity_results_per_method[method] = sensitivity_results


# Prepare data for sensitivity boxplots
sensitivity_boxplot_data = []

for method, sensitivity_results in sensitivity_results_per_method.items():
    for cell_type, sensitivities in sensitivity_results.items():
        method_df = pd.DataFrame({
            'Cell Type': cell_type,
            'Sensitivity': sensitivities,
            'Segmentation Method': method
        })
        sensitivity_boxplot_data.append(method_df)

# Concatenate all the dataframes into one
sensitivity_boxplot_data = pd.concat(sensitivity_boxplot_data)
    
plot_sensitivity_boxplots(sensitivity_boxplot_data, figures_path, palette=method_colors)