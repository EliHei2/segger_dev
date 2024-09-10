segger.validation.utils
=======================

.. py:module:: segger.validation.utils




Module Contents
---------------

.. py:function:: find_markers(adata: anndata.AnnData, cell_type_column: str, pos_percentile: float = 5, neg_percentile: float = 10, percentage: float = 50) -> Dict[str, Dict[str, List[str]]]

   Identify positive and negative markers for each cell type based on gene expression and filter by expression percentage.

   Parameters:
   - adata: AnnData
       Annotated data object containing gene expression data.
   - cell_type_column: str
       Column name in `adata.obs` that specifies cell types.
   - pos_percentile: float, default=5
       Percentile threshold to determine top x% expressed genes.
   - neg_percentile: float, default=10
       Percentile threshold to determine top x% lowly expressed genes.
   - percentage: float, default=50
       Minimum percentage of cells expressing the marker within a cell type for it to be considered.

   Returns:
   - markers: dict
       Dictionary where keys are cell types and values are dictionaries containing:
           'positive': list of top x% highly expressed genes
           'negative': list of top x% lowly expressed genes.


.. py:function:: find_mutually_exclusive_genes(adata: anndata.AnnData, markers: Dict[str, Dict[str, List[str]]], cell_type_column: str) -> List[Tuple[str, str]]

   Identify mutually exclusive genes based on expression criteria.

   Parameters:
   - adata: AnnData
       Annotated data object containing gene expression data.
   - markers: dict
       Dictionary where keys are cell types and values are dictionaries containing:
           'positive': list of top x% highly expressed genes
           'negative': list of top x% lowly expressed genes.
   - cell_type_column: str
       Column name in `adata.obs` that specifies cell types.

   Returns:
   - exclusive_pairs: list
       List of mutually exclusive gene pairs.


.. py:function:: compute_MECR(adata: anndata.AnnData, gene_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]

   Compute the Mutually Exclusive Co-expression Rate (MECR) for each gene pair in an AnnData object.

   Parameters:
   - adata: AnnData
       Annotated data object containing gene expression data.
   - gene_pairs: List[Tuple[str, str]]
       List of tuples representing gene pairs to evaluate.

   Returns:
   - mecr_dict: Dict[Tuple[str, str], float]
       Dictionary where keys are gene pairs (tuples) and values are MECR values.


.. py:function:: compute_quantized_mecr_area(adata: scanpy.AnnData, gene_pairs: List[Tuple[str, str]], quantiles: int = 10) -> pandas.DataFrame

   Compute the average MECR, variance of MECR, and average cell area for quantiles of cell areas.

   Parameters:
   - adata: AnnData
       Annotated data object containing gene expression data.
   - gene_pairs: List[Tuple[str, str]]
       List of tuples representing gene pairs to evaluate.
   - quantiles: int, default=10
       Number of quantiles to divide the data into.

   Returns:
   - quantized_data: pd.DataFrame
       DataFrame containing quantile information, average MECR, variance of MECR, average area, and number of cells.


.. py:function:: compute_quantized_mecr_counts(adata: scanpy.AnnData, gene_pairs: List[Tuple[str, str]], quantiles: int = 10) -> pandas.DataFrame

   Compute the average MECR, variance of MECR, and average transcript counts for quantiles of transcript counts.

   Parameters:
   - adata: AnnData
       Annotated data object containing gene expression data.
   - gene_pairs: List[Tuple[str, str]]
       List of tuples representing gene pairs to evaluate.
   - quantiles: int, default=10
       Number of quantiles to divide the data into.

   Returns:
   - quantized_data: pd.DataFrame
       DataFrame containing quantile information, average MECR, variance of MECR, average counts, and number of cells.


.. py:function:: annotate_query_with_reference(reference_adata: anndata.AnnData, query_adata: anndata.AnnData, transfer_column: str) -> anndata.AnnData

   Annotate query AnnData object using a scRNA-seq reference atlas.

   Parameters:
   - reference_adata: ad.AnnData
       Reference AnnData object containing the scRNA-seq atlas data.
   - query_adata: ad.AnnData
       Query AnnData object containing the data to be annotated.
   - transfer_column: str
       The name of the column in the reference atlas's `obs` to transfer to the query dataset.

   Returns:
   - query_adata: ad.AnnData
       Annotated query AnnData object with transferred labels and UMAP coordinates from the reference.


.. py:function:: calculate_contamination(adata: anndata.AnnData, markers: Dict[str, Dict[str, List[str]]], radius: float = 15, n_neighs: int = 10, celltype_column: str = 'celltype_major', num_cells: int = 10000) -> pandas.DataFrame

   Calculate normalized contamination from neighboring cells of different cell types based on positive markers.

   Parameters:
   - adata: ad.AnnData
       Annotated data object with raw counts and cell type information.
   - markers: dict
       Dictionary where keys are cell types and values are dictionaries containing:
           'positive': list of top x% highly expressed genes
           'negative': list of top x% lowly expressed genes.
   - radius: float, default=15
       Radius for spatial neighbor calculation.
   - n_neighs: int, default=10
       Maximum number of neighbors to consider.
   - celltype_column: str, default='celltype_major'
       Column name in the AnnData object representing cell types.
   - num_cells: int, default=10000
       Number of cells to randomly select for the calculation.

   Returns:
   - contamination_df: pd.DataFrame
       DataFrame containing the normalized level of contamination from each cell type to each other cell type.


.. py:function:: calculate_sensitivity(adata: anndata.AnnData, purified_markers: Dict[str, List[str]], max_cells_per_type: int = 1000) -> Dict[str, List[float]]

   Calculate the sensitivity of the purified markers for each cell type.

   Parameters:
   - adata: AnnData
       Annotated data object containing gene expression data.
   - purified_markers: dict
       Dictionary where keys are cell types and values are lists of purified marker genes.
   - max_cells_per_type: int, default=1000
       Maximum number of cells to consider per cell type.

   Returns:
   - sensitivity_results: dict
       Dictionary with cell types as keys and lists of sensitivity values for each cell.


.. py:function:: compute_clustering_scores(adata: anndata.AnnData, cell_type_column: str = 'celltype_major', use_pca: bool = True) -> Tuple[float, float]

   Compute the Calinski-Harabasz and Silhouette scores for an AnnData object based on the assigned cell types.

   Parameters:
   - adata: AnnData
       Annotated data object containing gene expression data and cell type assignments.
   - cell_type_column: str, default='celltype_major'
       Column name in `adata.obs` that specifies cell types.
   - use_pca: bool, default=True
       Whether to use PCA components as features. If False, use the raw data.

   Returns:
   - ch_score: float
       The Calinski-Harabasz score.
   - sh_score: float
       The Silhouette score.


.. py:function:: compute_neighborhood_metrics(adata: anndata.AnnData, radius: float = 10, celltype_column: str = 'celltype_major', n_neighs: int = 20, subset_size: int = 10000) -> None

   Compute neighborhood entropy and number of neighbors for each cell in the AnnData object.

   Parameters:
   - adata: AnnData
       Annotated data object containing spatial information and cell type assignments.
   - radius: int, default=10
       Radius for spatial neighbor calculation.
   - celltype_column: str, default='celltype_major'
       Column name in `adata.obs` that specifies cell types.


.. py:function:: compute_transcript_density(adata: anndata.AnnData) -> None

   Compute the transcript density for each cell in the AnnData object.

   Parameters:
   - adata: AnnData
       Annotated data object containing transcript and cell area information.


.. py:function:: plot_metric_comparison(ax: matplotlib.pyplot.Axes, data: pandas.DataFrame, metric: str, label: str, method1: str, method2: str) -> None

   Plot a comparison of a specific metric between two methods.

   Parameters:
   - ax: plt.Axes
       Matplotlib axis to plot on.
   - data: pd.DataFrame
       DataFrame containing the data for plotting.
   - metric: str
       The metric to compare.
   - label: str
       Label for the metric.
   - method1: str
       The first method to compare.
   - method2: str
       The second method to compare.


.. py:function:: load_segmentations(segmentation_paths: Dict[str, pathlib.Path]) -> Dict[str, scanpy.AnnData]

   Load segmentation data from provided paths and handle special cases like separating 'segger' into 'segger_n0' and 'segger_n1'.

   Parameters:
   segmentation_paths (Dict[str, Path]): Dictionary mapping segmentation method names to their file paths.

   Returns:
   Dict[str, sc.AnnData]: Dictionary mapping segmentation method names to loaded AnnData objects.


.. py:function:: plot_cell_counts(segmentations_dict: Dict[str, scanpy.AnnData], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot the number of cells per segmentation method.

   Parameters:
   segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
   output_path (Path): Path to the directory where the plot will be saved.


.. py:function:: plot_percent_assigned(segmentations_dict: Dict[str, scanpy.AnnData], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot the percentage of assigned transcripts (normalized) for each segmentation method.

   Parameters:
   segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
   output_path (Path): Path to the directory where the plot will be saved.


.. py:function:: plot_gene_counts(segmentations_dict: Dict[str, scanpy.AnnData], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot the normalized gene counts for each segmentation method.

   Parameters:
   segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
   output_path (Path): Path to the directory where the plot will be saved.


.. py:function:: plot_counts_per_cell(segmentations_dict: Dict[str, scanpy.AnnData], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot the counts per cell (log2) for each segmentation method.

   Parameters:
   segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
   output_path (Path): Path to the directory where the plot will be saved.


.. py:function:: plot_cell_area(segmentations_dict: Dict[str, scanpy.AnnData], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot the cell area (log2) for each segmentation method.

   Parameters:
   segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
   output_path (Path): Path to the directory where the plot will be saved.


.. py:function:: plot_transcript_density(segmentations_dict: Dict[str, scanpy.AnnData], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot the transcript density (log2) for each segmentation method.

   Parameters:
   segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
   output_path (Path): Path to the directory where the plot will be saved.


.. py:function:: plot_general_statistics_plots(segmentations_dict: Dict[str, scanpy.AnnData], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Create a summary plot with all the general statistics subplots.

   Parameters:
   segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
   output_path (Path): Path to the directory where the summary plot will be saved.


.. py:function:: plot_mecr_results(mecr_results: Dict[str, Dict[Tuple[str, str], float]], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot the MECR (Mutually Exclusive Co-expression Rate) results for each segmentation method.

   Parameters:
   mecr_results (Dict[str, Dict[Tuple[str, str], float]]): Dictionary of MECR results for each segmentation method.
   output_path (Path): Path to the directory where the plot will be saved.
   palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.


.. py:function:: plot_quantized_mecr_counts(quantized_mecr_counts: Dict[str, pandas.DataFrame], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot the quantized MECR values against transcript counts for each segmentation method, with point size proportional to the variance of MECR.

   Parameters:
   quantized_mecr_counts (Dict[str, pd.DataFrame]): Dictionary of quantized MECR count data for each segmentation method.
   output_path (Path): Path to the directory where the plot will be saved.
   palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.


.. py:function:: plot_quantized_mecr_area(quantized_mecr_area: Dict[str, pandas.DataFrame], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot the quantized MECR values against cell areas for each segmentation method, with point size proportional to the variance of MECR.

   Parameters:
   quantized_mecr_area (Dict[str, pd.DataFrame]): Dictionary of quantized MECR area data for each segmentation method.
   output_path (Path): Path to the directory where the plot will be saved.
   palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.


.. py:function:: plot_contamination_results(contamination_results: Dict[str, pandas.DataFrame], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot contamination results for each segmentation method.

   Parameters:
   contamination_results (Dict[str, pd.DataFrame]): Dictionary of contamination data for each segmentation method.
   output_path (Path): Path to the directory where the plot will be saved.
   palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.


.. py:function:: plot_contamination_boxplots(boxplot_data: pandas.DataFrame, output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot boxplots for contamination values across different segmentation methods.

   Parameters:
   boxplot_data (pd.DataFrame): DataFrame containing contamination data for all segmentation methods.
   output_path (Path): Path to the directory where the plot will be saved.
   palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.


.. py:function:: plot_umaps_with_scores(segmentations_dict: Dict[str, scanpy.AnnData], clustering_scores: Dict[str, Tuple[float, float]], output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot UMAPs colored by cell type for each segmentation method and display clustering scores in the title.
   Parameters:
   segmentations_dict (Dict[str, AnnData]): Dictionary of AnnData objects for each segmentation method.
   clustering_scores (Dict[str, Tuple[float, float]]): Dictionary of clustering scores for each method.
   output_path (Path): Path to the directory where the plots will be saved.
   palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.


.. py:function:: plot_entropy_boxplots(entropy_boxplot_data: pandas.DataFrame, output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot boxplots for neighborhood entropy across different segmentation methods by cell type.

   Parameters:
   entropy_boxplot_data (pd.DataFrame): DataFrame containing neighborhood entropy data for all segmentation methods.
   output_path (Path): Path to the directory where the plot will be saved.
   palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.


.. py:function:: plot_sensitivity_boxplots(sensitivity_boxplot_data: pandas.DataFrame, output_path: pathlib.Path, palette: Dict[str, str]) -> None

   Plot boxplots for sensitivity across different segmentation methods by cell type.
   Parameters:
   sensitivity_boxplot_data (pd.DataFrame): DataFrame containing sensitivity data for all segmentation methods.
   output_path (Path): Path to the directory where the plot will be saved.
   palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.


