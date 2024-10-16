import pandas as pd
import numpy as np
import anndata as ad
from scipy.spatial import ConvexHull
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import scanpy as sc
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import dask

dask.config.set({"dataframe.query-planning": False})
import squidpy as sq
from sklearn.metrics import calinski_harabasz_score, silhouette_score, f1_score
from pathlib import Path
import seaborn as sns


def find_markers(
    adata: ad.AnnData,
    cell_type_column: str,
    pos_percentile: float = 5,
    neg_percentile: float = 10,
    percentage: float = 50,
) -> Dict[str, Dict[str, List[str]]]:
    """Identify positive and negative markers for each cell type based on gene expression and filter by expression percentage.

    Args:
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
    """
    markers = {}
    sc.tl.rank_genes_groups(adata, groupby=cell_type_column)
    genes = adata.var_names
    for cell_type in adata.obs[cell_type_column].unique():
        subset = adata[adata.obs[cell_type_column] == cell_type]
        mean_expression = np.asarray(subset.X.mean(axis=0)).flatten()
        cutoff_high = np.percentile(mean_expression, 100 - pos_percentile)
        cutoff_low = np.percentile(mean_expression, neg_percentile)
        pos_indices = np.where(mean_expression >= cutoff_high)[0]
        neg_indices = np.where(mean_expression <= cutoff_low)[0]
        expr_frac = np.asarray((subset.X[:, pos_indices] > 0).mean(axis=0)).flatten()
        valid_pos_indices = pos_indices[expr_frac >= (percentage / 100)]
        positive_markers = genes[valid_pos_indices]
        negative_markers = genes[neg_indices]
        markers[cell_type] = {"positive": list(positive_markers), "negative": list(negative_markers)}
    return markers


def find_mutually_exclusive_genes(
    adata: ad.AnnData, markers: Dict[str, Dict[str, List[str]]], cell_type_column: str
) -> List[Tuple[str, str]]:
    """Identify mutually exclusive genes based on expression criteria.

    Args:
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
    """
    exclusive_genes = {}
    all_exclusive = []
    gene_expression = adata.to_df()
    for cell_type, marker_sets in markers.items():
        positive_markers = marker_sets["positive"]
        exclusive_genes[cell_type] = []
        for gene in positive_markers:
            gene_expr = adata[:, gene].X
            cell_type_mask = adata.obs[cell_type_column] == cell_type
            non_cell_type_mask = ~cell_type_mask
            if (gene_expr[cell_type_mask] > 0).mean() > 0.2 and (gene_expr[non_cell_type_mask] > 0).mean() < 0.05:
                exclusive_genes[cell_type].append(gene)
                all_exclusive.append(gene)
    unique_genes = list({gene for i in exclusive_genes.keys() for gene in exclusive_genes[i] if gene in all_exclusive})
    filtered_exclusive_genes = {
        i: [gene for gene in exclusive_genes[i] if gene in unique_genes] for i in exclusive_genes.keys()
    }
    mutually_exclusive_gene_pairs = [
        (gene1, gene2)
        for key1, key2 in combinations(filtered_exclusive_genes.keys(), 2)
        for gene1 in filtered_exclusive_genes[key1]
        for gene2 in filtered_exclusive_genes[key2]
    ]
    return mutually_exclusive_gene_pairs


def compute_MECR(adata: ad.AnnData, gene_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    """Compute the Mutually Exclusive Co-expression Rate (MECR) for each gene pair in an AnnData object.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - gene_pairs: List[Tuple[str, str]]
        List of tuples representing gene pairs to evaluate.

    Returns:
    - mecr_dict: Dict[Tuple[str, str], float]
        Dictionary where keys are gene pairs (tuples) and values are MECR values.
    """
    mecr_dict = {}
    gene_expression = adata.to_df()
    for gene1, gene2 in gene_pairs:
        expr_gene1 = gene_expression[gene1] > 0
        expr_gene2 = gene_expression[gene2] > 0
        both_expressed = (expr_gene1 & expr_gene2).mean()
        at_least_one_expressed = (expr_gene1 | expr_gene2).mean()
        mecr = both_expressed / at_least_one_expressed if at_least_one_expressed > 0 else 0
        mecr_dict[(gene1, gene2)] = mecr
    return mecr_dict


def compute_quantized_mecr_area(
    adata: sc.AnnData, gene_pairs: List[Tuple[str, str]], quantiles: int = 10
) -> pd.DataFrame:
    """Compute the average MECR, variance of MECR, and average cell area for quantiles of cell areas.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - gene_pairs: List[Tuple[str, str]]
        List of tuples representing gene pairs to evaluate.
    - quantiles: int, default=10
        Number of quantiles to divide the data into.

    Returns:
    - quantized_data: pd.DataFrame
        DataFrame containing quantile information, average MECR, variance of MECR, average area, and number of cells.
    """
    adata.obs["quantile"] = pd.qcut(adata.obs["cell_area"], quantiles, labels=False)
    quantized_data = []
    for quantile in range(quantiles):
        cells_in_quantile = adata.obs["quantile"] == quantile
        mecr = compute_MECR(adata[cells_in_quantile, :], gene_pairs)
        average_mecr = np.mean([i for i in mecr.values()])
        variance_mecr = np.var([i for i in mecr.values()])
        average_area = adata.obs.loc[cells_in_quantile, "cell_area"].mean()
        quantized_data.append(
            {
                "quantile": quantile / quantiles,
                "average_mecr": average_mecr,
                "variance_mecr": variance_mecr,
                "average_area": average_area,
                "num_cells": cells_in_quantile.sum(),
            }
        )
    return pd.DataFrame(quantized_data)


def compute_quantized_mecr_counts(
    adata: sc.AnnData, gene_pairs: List[Tuple[str, str]], quantiles: int = 10
) -> pd.DataFrame:
    """Compute the average MECR, variance of MECR, and average transcript counts for quantiles of transcript counts.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - gene_pairs: List[Tuple[str, str]]
        List of tuples representing gene pairs to evaluate.
    - quantiles: int, default=10
        Number of quantiles to divide the data into.

    Returns:
    - quantized_data: pd.DataFrame
        DataFrame containing quantile information, average MECR, variance of MECR, average counts, and number of cells.
    """
    adata.obs["quantile"] = pd.qcut(adata.obs["transcripts"], quantiles, labels=False)
    quantized_data = []
    for quantile in range(quantiles):
        cells_in_quantile = adata.obs["quantile"] == quantile
        mecr = compute_MECR(adata[cells_in_quantile, :], gene_pairs)
        average_mecr = np.mean([i for i in mecr.values()])
        variance_mecr = np.var([i for i in mecr.values()])
        average_counts = adata.obs.loc[cells_in_quantile, "transcripts"].mean()
        quantized_data.append(
            {
                "quantile": quantile / quantiles,
                "average_mecr": average_mecr,
                "variance_mecr": variance_mecr,
                "average_counts": average_counts,
                "num_cells": cells_in_quantile.sum(),
            }
        )
    return pd.DataFrame(quantized_data)


def annotate_query_with_reference(
    reference_adata: ad.AnnData, query_adata: ad.AnnData, transfer_column: str
) -> ad.AnnData:
    """Annotate query AnnData object using a scRNA-seq reference atlas.

    Args:
    - reference_adata: ad.AnnData
        Reference AnnData object containing the scRNA-seq atlas data.
    - query_adata: ad.AnnData
        Query AnnData object containing the data to be annotated.
    - transfer_column: str
        The name of the column in the reference atlas's `obs` to transfer to the query dataset.

    Returns:
    - query_adata: ad.AnnData
        Annotated query AnnData object with transferred labels and UMAP coordinates from the reference.
    """
    common_genes = list(set(reference_adata.var_names) & set(query_adata.var_names))
    reference_adata = reference_adata[:, common_genes]
    query_adata = query_adata[:, common_genes]
    query_adata.layers["raw"] = query_adata.raw.X if query_adata.raw else query_adata.X
    query_adata.var["raw_counts"] = query_adata.layers["raw"].sum(axis=0)
    sc.pp.normalize_total(query_adata, target_sum=1e4)
    sc.pp.log1p(query_adata)
    sc.pp.pca(reference_adata)
    sc.pp.neighbors(reference_adata)
    sc.tl.umap(reference_adata)
    sc.tl.ingest(query_adata, reference_adata, obs=transfer_column)
    query_adata.obsm["X_umap"] = query_adata.obsm["X_umap"]
    return query_adata


def calculate_contamination(
    adata: ad.AnnData,
    markers: Dict[str, Dict[str, List[str]]],
    radius: float = 15,
    n_neighs: int = 10,
    celltype_column: str = "celltype_major",
    num_cells: int = 10000,
) -> pd.DataFrame:
    """Calculate normalized contamination from neighboring cells of different cell types based on positive markers.

    Args:
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
    """
    if celltype_column not in adata.obs:
        raise ValueError("Column celltype_column must be present in adata.obs.")
    positive_markers = {ct: markers[ct]["positive"] for ct in markers}
    adata.obsm["spatial"] = adata.obs[["cell_centroid_x", "cell_centroid_y"]].copy().to_numpy()
    sq.gr.spatial_neighbors(adata, radius=radius, n_neighs=n_neighs, coord_type="generic")
    neighbors = adata.obsp["spatial_connectivities"].tolil()
    raw_counts = adata[:, adata.var_names].layers["raw"].toarray()
    cell_types = adata.obs[celltype_column]
    selected_cells = np.random.choice(adata.n_obs, size=min(num_cells, adata.n_obs), replace=False)
    contamination = {ct: {ct2: 0 for ct2 in positive_markers.keys()} for ct in positive_markers.keys()}
    negighborings = {ct: {ct2: 0 for ct2 in positive_markers.keys()} for ct in positive_markers.keys()}
    for cell_idx in selected_cells:
        cell_type = cell_types[cell_idx]
        own_markers = set(positive_markers[cell_type])
        for marker in own_markers:
            if marker in adata.var_names:
                total_counts_in_neighborhood = raw_counts[cell_idx, adata.var_names.get_loc(marker)]
                for neighbor_idx in neighbors.rows[cell_idx]:
                    total_counts_in_neighborhood += raw_counts[neighbor_idx, adata.var_names.get_loc(marker)]
                for neighbor_idx in neighbors.rows[cell_idx]:
                    neighbor_type = cell_types[neighbor_idx]
                    if cell_type == neighbor_type:
                        continue
                    neighbor_markers = set(positive_markers.get(neighbor_type, []))
                    contamination_markers = own_markers - neighbor_markers
                    for marker in contamination_markers:
                        if marker in adata.var_names:
                            marker_counts_in_neighbor = raw_counts[neighbor_idx, adata.var_names.get_loc(marker)]
                            if total_counts_in_neighborhood > 0:
                                contamination[cell_type][neighbor_type] += (
                                    marker_counts_in_neighbor / total_counts_in_neighborhood
                                )
                                negighborings[cell_type][neighbor_type] += 1
    contamination_df = pd.DataFrame(contamination).T
    negighborings_df = pd.DataFrame(negighborings).T
    contamination_df.index.name = "Source Cell Type"
    contamination_df.columns.name = "Target Cell Type"
    return contamination_df / (negighborings_df + 1)


def calculate_sensitivity(
    adata: ad.AnnData, purified_markers: Dict[str, List[str]], max_cells_per_type: int = 1000
) -> Dict[str, List[float]]:
    """Calculate the sensitivity of the purified markers for each cell type.

    Args:
    - adata: AnnData
        Annotated data object containing gene expression data.
    - purified_markers: dict
        Dictionary where keys are cell types and values are lists of purified marker genes.
    - max_cells_per_type: int, default=1000
        Maximum number of cells to consider per cell type.

    Returns:
    - sensitivity_results: dict
        Dictionary with cell types as keys and lists of sensitivity values for each cell.
    """
    sensitivity_results = {cell_type: [] for cell_type in purified_markers.keys()}
    for cell_type, markers in purified_markers.items():
        markers = markers["positive"]
        subset = adata[adata.obs["celltype_major"] == cell_type]
        if subset.n_obs > max_cells_per_type:
            cell_indices = np.random.choice(subset.n_obs, max_cells_per_type, replace=False)
            subset = subset[cell_indices]
        for cell_counts in subset.X:
            expressed_markers = np.asarray((cell_counts[subset.var_names.get_indexer(markers)] > 0).sum())
            sensitivity = expressed_markers / len(markers) if markers else 0
            sensitivity_results[cell_type].append(sensitivity)
    return sensitivity_results


def compute_clustering_scores(
    adata: ad.AnnData, cell_type_column: str = "celltype_major", use_pca: bool = True
) -> Tuple[float, float]:
    """Compute the Calinski-Harabasz and Silhouette scores for an AnnData object based on the assigned cell types.

    Args:
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
    """
    if cell_type_column not in adata.obs:
        raise ValueError(f"Column '{cell_type_column}' must be present in adata.obs.")
    features = adata.X
    cell_indices = np.random.choice(adata.n_obs, 10000, replace=False)
    features = features[cell_indices, :]
    labels = adata[cell_indices, :].obs[cell_type_column]
    ch_score = calinski_harabasz_score(features, labels)
    sh_score = silhouette_score(features, labels)
    return ch_score, sh_score


def compute_neighborhood_metrics(
    adata: ad.AnnData,
    radius: float = 10,
    celltype_column: str = "celltype_major",
    n_neighs: int = 20,
    subset_size: int = 10000,
) -> None:
    """Compute neighborhood entropy and number of neighbors for each cell in the AnnData object.

    Args:
    - adata: AnnData
        Annotated data object containing spatial information and cell type assignments.
    - radius: int, default=10
        Radius for spatial neighbor calculation.
    - celltype_column: str, default='celltype_major'
        Column name in `adata.obs` that specifies cell types.
    """
    """
    Compute neighborhood entropy and number of neighbors for a random subset of cells in the AnnData object.

    Args:
    - adata: AnnData
        Annotated data object containing spatial information and cell type assignments.
    - radius: int, default=10
        Radius for spatial neighbor calculation.
    - celltype_column: str, default='celltype_major'
        Column name in `adata.obs` that specifies cell types.
    - subset_size: int, default=10000
        Number of cells to randomly select for the calculation.
    """
    # Ensure the subset size does not exceed the number of cells
    subset_size = min(subset_size, adata.n_obs)
    # Randomly select a subset of cells
    subset_indices = np.random.choice(adata.n_obs, subset_size, replace=False)
    # Compute spatial neighbors for the entire dataset
    sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic", n_neighs=n_neighs)
    neighbors = adata.obsp["spatial_distances"].tolil().rows
    entropies = []
    num_neighbors = []
    # Calculate entropy and number of neighbors only for the selected subset
    for cell_index in subset_indices:
        neighbor_indices = neighbors[cell_index]
        neighbor_cell_types = adata.obs[celltype_column].iloc[neighbor_indices]
        cell_type_counts = neighbor_cell_types.value_counts()
        total_neighbors = len(neighbor_cell_types)
        num_neighbors.append(total_neighbors)
        if total_neighbors > 0:
            cell_type_probs = cell_type_counts / total_neighbors
            cell_type_entropy = entropy(cell_type_probs)
            entropies.append(cell_type_entropy)
        else:
            entropies.append(0)
    # Store the results back into the original AnnData object
    # We fill with NaN for cells not in the subset
    entropy_full = np.full(adata.n_obs, np.nan)
    neighbors_full = np.full(adata.n_obs, np.nan)
    entropy_full[subset_indices] = entropies
    neighbors_full[subset_indices] = num_neighbors
    adata.obs["neighborhood_entropy"] = entropy_full
    adata.obs["number_of_neighbors"] = neighbors_full


def compute_transcript_density(adata: ad.AnnData) -> None:
    """Compute the transcript density for each cell in the AnnData object.

    Args:
    - adata: AnnData
        Annotated data object containing transcript and cell area information.
    """
    try:
        transcript_counts = adata.obs["transcript_counts"]
    except:
        transcript_counts = adata.obs["transcripts"]
    cell_areas = adata.obs["cell_area"]
    adata.obs["transcript_density"] = transcript_counts / cell_areas


# def compute_celltype_f1_purity(
#     adata: ad.AnnData,
#     marker_genes: Dict[str, Dict[str, List[str]]]
# ) -> Dict[str, float]:
#     """
#     Compute the purity F1 score for each cell type based on marker genes.

#     Args:
#     - adata: AnnData
#         Annotated data object containing gene expression data.
#     - marker_genes: dict
#         Dictionary where keys are cell types and values are dictionaries containing:
#             'positive': list of top x% highly expressed genes
#             'negative': list of top x% lowly expressed genes.

#     Returns:
#     - f1_scores: dict
#         Dictionary with cell types as keys and their corresponding purity F1 scores.
#     """
#     f1_scores = {}
#     for cell_type, markers in marker_genes.items():
#         pos_markers = markers['positive']
#         neg_markers = markers['negative']
#         pos_expr = adata[:, pos_markers].X.toarray().mean(axis=1)
#         neg_expr = adata[:, neg_markers].X.toarray().mean(axis=1)
#         pos_labels = adata.obs['celltype_major'] == cell_type
#         neg_labels = adata.obs['celltype_major'] != cell_type
#         pos_f1 = f1_score(pos_labels, pos_expr >= np.percentile(pos_expr, 97))
#         neg_f1 = f1_score(pos_labels, neg_expr <= np.percentile(neg_expr, 10))
#         scaled_pos_f1 = (pos_f1 - 0) / (1 - 0)
#         scaled_neg_f1 = (neg_f1 - 0) / (1 - 0)
#         purity_f1 = 2 * (scaled_pos_f1 * scaled_neg_f1) / (scaled_pos_f1 + scaled_neg_f1)
#         f1_scores[cell_type] = purity_f1
#     return f1_scores


# def average_log_normalized_expression(
#     adata: ad.AnnData,
#     celltype_column: str
# ) -> pd.DataFrame:
#     """
#     Compute the average log-normalized expression for each cell type.

#     Args:
#     - adata: AnnData
#         Annotated data object containing gene expression data.
#     - celltype_column: str
#         Column name in `adata.obs` that specifies cell types.

#     Returns:
#     - avg_expr: pd.DataFrame
#         DataFrame containing the average log-normalized expression for each cell type.
#     """
#     return adata.to_df().groupby(adata.obs[celltype_column]).mean()


def plot_metric_comparison(
    ax: plt.Axes, data: pd.DataFrame, metric: str, label: str, method1: str, method2: str, output_path: Path
) -> None:
    """Plot a comparison of a specific metric between two methods and save the comparison data.

    Args:
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
    - output_path: Path
        Path to save the merged DataFrame as a CSV.
    """
    subset1 = data[data["method"] == method1]
    subset2 = data[data["method"] == method2]
    merged_data = pd.merge(subset1, subset2, on="celltype_major", suffixes=(f"_{method1}", f"_{method2}"))

    # Save the merged data used in the plot to CSV
    merged_data.to_csv(output_path / f"metric_comparison_{metric}_{method1}_vs_{method2}.csv", index=False)

    for cell_type in merged_data["celltype_major"].unique():
        cell_data = merged_data[merged_data["celltype_major"] == cell_type]
        ax.scatter(cell_data[f"{metric}_{method1}"], cell_data[f"{metric}_{method2}"], label=cell_type)

    max_value = max(merged_data[f"{metric}_{method1}"].max(), merged_data[f"{metric}_{method2}"].max())
    ax.plot([0, max_value], [0, max_value], "k--", alpha=0.5)
    ax.set_xlabel(f"{label} ({method1})")
    ax.set_ylabel(f"{label} ({method2})")
    ax.set_title(f"{label}: {method1} vs {method2}")


def load_segmentations(segmentation_paths: Dict[str, Path]) -> Dict[str, sc.AnnData]:
    """Load segmentation data from provided paths and handle special cases like separating 'segger' into 'segger_n0' and 'segger_n1'.

    Args:
    segmentation_paths (Dict[str, Path]): Dictionary mapping segmentation method names to their file paths.

    Returns:
    Dict[str, sc.AnnData]: Dictionary mapping segmentation method names to loaded AnnData objects.
    """
    segmentations_dict = {}
    for method, path in segmentation_paths.items():
        adata = sc.read(path)
        # Special handling for 'segger' to separate into 'segger_n0' and 'segger_n1'
        if method == "segger":
            cells_n1 = [i for i in adata.obs_names if not i.endswith("-nx")]
            cells_n0 = [i for i in adata.obs_names if i.endswith("-nx")]
            segmentations_dict["segger_n1"] = adata[cells_n1, :]
            segmentations_dict["segger_n0"] = adata[cells_n0, :]
        segmentations_dict[method] = adata
    return segmentations_dict


def plot_cell_counts(segmentations_dict: Dict[str, sc.AnnData], output_path: Path, palette: Dict[str, str]) -> None:
    """Plot the number of cells per segmentation method and save the cell count data as a CSV.

    Args:
    segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
    output_path (Path): Path to the directory where the plot will be saved.
    """
    # Calculate the number of cells in each segmentation method
    cell_counts = {method: seg.n_obs for method, seg in segmentations_dict.items()}

    # Create a DataFrame for the bar plot
    df = pd.DataFrame(cell_counts, index=["Number of Cells"]).T

    # Save the DataFrame to CSV
    df.to_csv(output_path / "cell_counts_data.csv", index=True)

    # Generate the bar plot
    ax = df.plot(
        kind="bar", stacked=False, color=[palette.get(key, "#333333") for key in df.index], figsize=(3, 6), width=0.9
    )

    # Add a dashed line for the 10X baseline
    if "10X" in cell_counts:
        baseline_height = cell_counts["10X"]
        ax.axhline(y=baseline_height, color="gray", linestyle="--", linewidth=1.5, label="10X Baseline")

    # Set plot titles and labels
    plt.title("Number of Cells per Segmentation Method")
    plt.xlabel("Segmentation Method")
    plt.ylabel("Number of Cells")
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save the figure as a PDF
    plt.savefig(output_path / "cell_counts_bar_plot.pdf", bbox_inches="tight")
    plt.show()


def plot_percent_assigned(
    segmentations_dict: Dict[str, sc.AnnData], output_path: Path, palette: Dict[str, str]
) -> None:
    """Plot the percentage of assigned transcripts (normalized) for each segmentation method.

    Args:
    segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
    output_path (Path): Path to the directory where the plot will be saved.
    """
    # Calculate total counts per gene for each segmentation method
    total_counts_per_gene = pd.DataFrame()

    for method, adata in segmentations_dict.items():
        gene_counts = adata.X.sum(axis=0).flatten()  # Sum across cells for each gene and flatten to 1D
        gene_counts = pd.Series(gene_counts, index=adata.var_names, name=method)
        total_counts_per_gene = pd.concat([total_counts_per_gene, gene_counts], axis=1)

    # Normalize by the maximum count per gene across all segmentations
    max_counts_per_gene = total_counts_per_gene.max(axis=1)
    percent_assigned_normalized = total_counts_per_gene.divide(max_counts_per_gene, axis=0) * 100

    # Prepare the data for the violin plot
    violin_data = pd.DataFrame({"Segmentation Method": [], "Percent Assigned (Normalized)": []})

    # Add normalized percent_assigned data for each method
    for method in segmentations_dict.keys():
        method_data = percent_assigned_normalized[method].dropna()
        method_df = pd.DataFrame(
            {"Segmentation Method": [method] * len(method_data), "Percent Assigned (Normalized)": method_data.values}
        )
        violin_data = pd.concat([violin_data, method_df], axis=0)

    violin_data.to_csv(output_path / "percent_assigned_normalized.csv", index=True)

    # Plot the violin plots
    plt.figure(figsize=(12, 8))
    ax = sns.violinplot(x="Segmentation Method", y="Percent Assigned (Normalized)", data=violin_data, palette=palette)

    # Add a dashed line for the 10X baseline
    if "10X" in segmentations_dict:
        baseline_height = percent_assigned_normalized["10X"].mean()
        ax.axhline(y=baseline_height, color="gray", linestyle="--", linewidth=1.5, label="10X Baseline")

    # Set plot titles and labels
    plt.title("")
    plt.xlabel("Segmentation Method")
    plt.ylabel("Percent Assigned (Normalized)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save the figure as a PDF
    plt.savefig(output_path / "percent_assigned_normalized_violin_plot.pdf", bbox_inches="tight")
    plt.show()


def plot_gene_counts(segmentations_dict: Dict[str, sc.AnnData], output_path: Path, palette: Dict[str, str]) -> None:
    """Plot the normalized gene counts for each segmentation method.

    Args:
    segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
    output_path (Path): Path to the directory where the plot will be saved.
    """
    # Calculate total counts per gene for each segmentation method
    total_counts_per_gene = pd.DataFrame()

    for method, adata in segmentations_dict.items():
        gene_counts = adata.X.sum(axis=0).flatten()
        gene_counts = pd.Series(gene_counts, index=adata.var_names, name=method)
        total_counts_per_gene = pd.concat([total_counts_per_gene, gene_counts], axis=1)

    # Normalize by the maximum count per gene across all segmentations
    max_counts_per_gene = total_counts_per_gene.max(axis=1)
    normalized_counts_per_gene = total_counts_per_gene.divide(max_counts_per_gene, axis=0)

    # Prepare the data for the box plot
    boxplot_data = pd.DataFrame({"Segmentation Method": [], "Normalized Counts": []})

    for method in segmentations_dict.keys():
        method_counts = normalized_counts_per_gene[method]
        method_df = pd.DataFrame(
            {"Segmentation Method": [method] * len(method_counts), "Normalized Counts": method_counts.values}
        )
        boxplot_data = pd.concat([boxplot_data, method_df], axis=0)

    boxplot_data.to_csv(output_path / "gene_counts_normalized_data.csv", index=True)

    # Plot the box plots
    plt.figure(figsize=(3, 6))
    ax = sns.boxplot(x="Segmentation Method", y="Normalized Counts", data=boxplot_data, palette=palette, width=0.9)

    # Add a dashed line for the 10X baseline
    if "10X" in normalized_counts_per_gene:
        baseline_height = normalized_counts_per_gene["10X"].mean()
        plt.axhline(y=baseline_height, color="gray", linestyle="--", linewidth=1.5, label="10X Baseline")

    # Set plot titles and labels
    plt.title("")
    plt.xlabel("Segmentation Method")
    plt.ylabel("Normalized Counts")
    plt.xticks(rotation=0)

    # Save the figure as a PDF
    plt.savefig(output_path / "gene_counts_normalized_boxplot_by_method.pdf", bbox_inches="tight")
    plt.show()


def plot_counts_per_cell(segmentations_dict: Dict[str, sc.AnnData], output_path: Path, palette: Dict[str, str]) -> None:
    """Plot the counts per cell (log2) for each segmentation method.

    Args:
    segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
    output_path (Path): Path to the directory where the plot will be saved.
    """
    # Prepare the data for the violin plot
    violin_data = pd.DataFrame({"Segmentation Method": [], "Counts per Cell (log2)": []})
    for method, adata in segmentations_dict.items():
        method_counts = adata.obs["transcripts"] + 1
        method_df = pd.DataFrame(
            {"Segmentation Method": [method] * len(method_counts), "Counts per Cell (log2)": method_counts.values}
        )
        violin_data = pd.concat([violin_data, method_df], axis=0)

    violin_data.to_csv(output_path / "counts_per_cell_data.csv", index=True)
    # Plot the violin plots
    plt.figure(figsize=(4, 6))
    ax = sns.violinplot(x="Segmentation Method", y="Counts per Cell (log2)", data=violin_data, palette=palette)
    ax.set(ylim=(5, 300))
    # Add a dashed line for the 10X-nucleus median
    if "10X-nucleus" in segmentations_dict:
        median_10X_nucleus = np.median(segmentations_dict["10X-nucleus"].obs["transcripts"] + 1)
        ax.axhline(y=median_10X_nucleus, color="gray", linestyle="--", linewidth=1.5, label="10X-nucleus Median")
    # Set plot titles and labels
    plt.title("")
    plt.xlabel("Segmentation Method")
    plt.ylabel("Counts per Cell (log2)")
    plt.xticks(rotation=0)
    # Save the figure as a PDF
    plt.savefig(output_path / "counts_per_cell_violin_plot.pdf", bbox_inches="tight")
    plt.show()


def plot_cell_area(segmentations_dict: Dict[str, sc.AnnData], output_path: Path, palette: Dict[str, str]) -> None:
    """Plot the cell area (log2) for each segmentation method.

    Args:
    segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
    output_path (Path): Path to the directory where the plot will be saved.
    """
    # Prepare the data for the violin plot
    violin_data = pd.DataFrame({"Segmentation Method": [], "Cell Area (log2)": []})
    for method in segmentations_dict.keys():
        if "cell_area" in segmentations_dict[method].obs.columns:
            method_area = segmentations_dict[method].obs["cell_area"] + 1
            method_df = pd.DataFrame(
                {"Segmentation Method": [method] * len(method_area), "Cell Area (log2)": method_area.values}
            )
            violin_data = pd.concat([violin_data, method_df], axis=0)
    violin_data.to_csv(output_path / "cell_area_log2_data.csv", index=True)
    # Plot the violin plots
    plt.figure(figsize=(4, 6))
    ax = sns.violinplot(x="Segmentation Method", y="Cell Area (log2)", data=violin_data, palette=palette)
    ax.set(ylim=(5, 100))
    # Add a dashed line for the 10X-nucleus median
    if "10X-nucleus" in segmentations_dict:
        median_10X_nucleus_area = np.median(segmentations_dict["10X-nucleus"].obs["cell_area"] + 1)
        ax.axhline(y=median_10X_nucleus_area, color="gray", linestyle="--", linewidth=1.5, label="10X-nucleus Median")
    # Set plot titles and labels
    plt.title("")
    plt.xlabel("Segmentation Method")
    plt.ylabel("Cell Area (log2)")
    plt.xticks(rotation=0)
    # Save the figure as a PDF
    plt.savefig(output_path / "cell_area_log2_violin_plot.pdf", bbox_inches="tight")
    plt.show()


def plot_transcript_density(
    segmentations_dict: Dict[str, sc.AnnData], output_path: Path, palette: Dict[str, str]
) -> None:
    """Plot the transcript density (log2) for each segmentation method.

    Args:
    segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
    output_path (Path): Path to the directory where the plot will be saved.
    """
    # Prepare the data for the violin plot
    violin_data = pd.DataFrame({"Segmentation Method": [], "Transcript Density (log2)": []})

    for method in segmentations_dict.keys():
        if "cell_area" in segmentations_dict[method].obs.columns:
            method_density = segmentations_dict[method].obs["transcripts"] / segmentations_dict[method].obs["cell_area"]
            method_density_log2 = np.log2(method_density + 1)
            method_df = pd.DataFrame(
                {
                    "Segmentation Method": [method] * len(method_density_log2),
                    "Transcript Density (log2)": method_density_log2.values,
                }
            )
            violin_data = pd.concat([violin_data, method_df], axis=0)

    violin_data.to_csv(output_path / "transcript_density_log2_data.csv", index=True)

    # Plot the violin plots
    plt.figure(figsize=(4, 6))
    ax = sns.violinplot(x="Segmentation Method", y="Transcript Density (log2)", data=violin_data, palette=palette)

    # Add a dashed line for the 10X-nucleus median
    if "10X-nucleus" in segmentations_dict:
        median_10X_nucleus_density_log2 = np.median(
            np.log2(
                segmentations_dict["10X-nucleus"].obs["transcripts"]
                / segmentations_dict["10X-nucleus"].obs["cell_area"]
                + 1
            )
        )
        ax.axhline(
            y=median_10X_nucleus_density_log2, color="gray", linestyle="--", linewidth=1.5, label="10X-nucleus Median"
        )

    # Set plot titles and labels
    plt.title("")
    plt.xlabel("Segmentation Method")
    plt.ylabel("Transcript Density (log2)")
    plt.xticks(rotation=0)

    # Save the figure as a PDF
    plt.savefig(output_path / "transcript_density_log2_violin_plot.pdf", bbox_inches="tight")
    plt.show()


def plot_general_statistics_plots(
    segmentations_dict: Dict[str, sc.AnnData], output_path: Path, palette: Dict[str, str]
) -> None:
    """Create a summary plot with all the general statistics subplots.

    Args:
    segmentations_dict (Dict[str, sc.AnnData]): Dictionary mapping segmentation method names to loaded AnnData objects.
    output_path (Path): Path to the directory where the summary plot will be saved.
    """
    plt.figure(figsize=(15, 20))

    plt.subplot(3, 2, 1)
    plot_cell_counts(segmentations_dict, output_path, palette=palette)

    plt.subplot(3, 2, 2)
    plot_percent_assigned(segmentations_dict, output_path, palette=palette)

    plt.subplot(3, 2, 3)
    plot_gene_counts(segmentations_dict, output_path, palette=palette)

    plt.subplot(3, 2, 4)
    plot_counts_per_cell(segmentations_dict, output_path, palette=palette)
    plt.subplot(3, 2, 5)
    plot_cell_area(segmentations_dict, output_path, palette=palette)

    plt.subplot(3, 2, 6)
    plot_transcript_density(segmentations_dict, output_path, palette=palette)

    plt.tight_layout()
    plt.savefig(output_path / "general_statistics_plots.pdf", bbox_inches="tight")
    plt.show()


def plot_mecr_results(
    mecr_results: Dict[str, Dict[Tuple[str, str], float]], output_path: Path, palette: Dict[str, str]
) -> None:
    """Plot the MECR (Mutually Exclusive Co-expression Rate) results for each segmentation method.

    Args:
    mecr_results (Dict[str, Dict[Tuple[str, str], float]]): Dictionary of MECR results for each segmentation method.
    output_path (Path): Path to the directory where the plot will be saved.
    palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.
    """
    # Prepare the data for plotting
    plot_data = []
    for method, mecr_dict in mecr_results.items():
        for gene_pair, mecr_value in mecr_dict.items():
            plot_data.append(
                {"Segmentation Method": method, "Gene Pair": f"{gene_pair[0]} - {gene_pair[1]}", "MECR": mecr_value}
            )
    df = pd.DataFrame(plot_data)
    df.to_csv(output_path / "mcer_box.csv", index=True)
    plt.figure(figsize=(3, 6))
    sns.boxplot(x="Segmentation Method", y="MECR", data=df, palette=palette)
    plt.title("Mutually Exclusive Co-expression Rate (MECR)")
    plt.xlabel("Segmentation Method")
    plt.ylabel("MECR")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "mecr_results_boxplot.pdf", bbox_inches="tight")
    plt.show()


def plot_quantized_mecr_counts(
    quantized_mecr_counts: Dict[str, pd.DataFrame], output_path: Path, palette: Dict[str, str]
) -> None:
    """Plot the quantized MECR values against transcript counts for each segmentation method, with point size proportional to the variance of MECR.

    Args:
        quantized_mecr_counts (Dict[str, pd.DataFrame]): Dictionary of quantized MECR count data for each segmentation method.
        output_path (Path): Path to the directory where the plot will be saved.
        palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.
    """
    # quantized_mecr_counts.to_csv(output_path / 'quantized_mecr_counts.csv', index=True)
    plt.figure(figsize=(9, 6))
    for method, df in quantized_mecr_counts.items():
        plt.plot(
            df["average_counts"],
            df["average_mecr"],
            marker="o",
            linestyle="-",
            color=palette.get(method, "#333333"),
            label=method,
            markersize=0,  # No markers, only lines
        )
        plt.scatter(
            df["average_counts"],
            df["average_mecr"],
            s=df["variance_mecr"] * 1e5,  # Size of points based on the variance of MECR
            color=palette.get(method, "#333333"),
            alpha=0.7,  # Slight transparency for overlapping points
            edgecolor="w",  # White edge color for better visibility
            linewidth=0.5,  # Thin edge line
        )
    plt.title("Quantized MECR by Transcript Counts")
    plt.xlabel("Average Transcript Counts")
    plt.ylabel("Average MECR")
    # Place the legend outside the plot on the top right
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path / "quantized_mecr_counts_plot.pdf", bbox_inches="tight")
    plt.show()


def plot_quantized_mecr_area(
    quantized_mecr_area: Dict[str, pd.DataFrame], output_path: Path, palette: Dict[str, str]
) -> None:
    """Plot the quantized MECR values against cell areas for each segmentation method, with point size proportional to the variance of MECR.

    Args:
    quantized_mecr_area (Dict[str, pd.DataFrame]): Dictionary of quantized MECR area data for each segmentation method.
    output_path (Path): Path to the directory where the plot will be saved.
    palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.
    """
    # quantized_mecr_area.to_csv(output_path / 'quantized_mecr_area.csv', index=True)
    plt.figure(figsize=(6, 4))
    for method, df in quantized_mecr_area.items():
        plt.plot(
            df["average_area"],
            df["average_mecr"],
            marker="o",
            # s=df['variance_mecr']  * 1e5,
            linestyle="-",
            color=palette.get(method, "#333333"),
            label=method,
            markersize=0,
        )
        plt.scatter(
            df["average_area"],
            df["average_mecr"],
            s=df["variance_mecr"] * 1e5,  # Size of points based on the variance of MECR
            color=palette.get(method, "#333333"),
            alpha=0.7,  # Slight transparency for overlapping points
            edgecolor="w",  # White edge color for better visibility
            linewidth=0.5,  # Thin edge line
        )
    plt.title("Quantized MECR by Cell Area")
    plt.xlabel("Average Cell Area")
    plt.ylabel("Average MECR")
    # Place the legend outside the plot on the top right
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path / "quantized_mecr_area_plot.pdf", bbox_inches="tight")
    plt.show()


def plot_contamination_results(
    contamination_results: Dict[str, pd.DataFrame], output_path: Path, palette: Dict[str, str]
) -> None:
    """Plot contamination results for each segmentation method.

    Args:
        contamination_results (Dict[str, pd.DataFrame]): Dictionary of contamination data for each segmentation method.
        output_path (Path): Path to the directory where the plot will be saved.
        palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.
    """
    contamination_results.to_csv(output_path / "contamination_results.csv", index=True)
    for method, df in contamination_results.items():
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title(f"Contamination Matrix for {method}")
        plt.xlabel("Target Cell Type")
        plt.ylabel("Source Cell Type")
        plt.tight_layout()
        plt.savefig(output_path / f"{method}_contamination_matrix.pdf", bbox_inches="tight")
        plt.show()


def plot_contamination_boxplots(boxplot_data: pd.DataFrame, output_path: Path, palette: Dict[str, str]) -> None:
    """Plot boxplots for contamination values across different segmentation methods.

    Args:
    boxplot_data (pd.DataFrame): DataFrame containing contamination data for all segmentation methods.
    output_path (Path): Path to the directory where the plot will be saved.
    palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.
    """
    boxplot_data.to_csv(output_path / "contamination_box_results.csv", index=True)
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="Source Cell Type", y="Contamination", hue="Segmentation Method", data=boxplot_data, palette=palette)
    plt.title("Neighborhood Contamination")
    plt.xlabel("Source Cell Type")
    plt.ylabel("Contamination")
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path / "contamination_boxplots.pdf", bbox_inches="tight")
    plt.show()


def plot_umaps_with_scores(
    segmentations_dict: Dict[str, sc.AnnData],
    clustering_scores: Dict[str, Tuple[float, float]],
    output_path: Path,
    palette: Dict[str, str],
) -> None:
    """Plot UMAPs colored by cell type for each segmentation method and display clustering scores in the title.
    Args:
    segmentations_dict (Dict[str, AnnData]): Dictionary of AnnData objects for each segmentation method.
    clustering_scores (Dict[str, Tuple[float, float]]): Dictionary of clustering scores for each method.
    output_path (Path): Path to the directory where the plots will be saved.
    palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.
    """
    for method, adata in segmentations_dict.items():
        print(method)
        adata_copy = adata.copy()
        sc.pp.subsample(adata_copy, n_obs=10000)
        sc.pp.normalize_total(adata_copy)
        # Plot UMAP colored by cell type
        plt.figure(figsize=(8, 6))
        sc.pp.neighbors(adata_copy, n_neighbors=5)
        sc.tl.umap(adata_copy, spread=5)
        sc.pl.umap(adata_copy, color="celltype_major", palette=palette, show=False)
        # Add clustering scores to the title
        ch_score, sh_score = compute_clustering_scores(adata_copy, cell_type_column="celltype_major")
        plt.title(f"{method} - UMAP\nCalinski-Harabasz: {ch_score:.2f}, Silhouette: {sh_score:.2f}")
        # Save the figure
        plt.savefig(output_path / f"{method}_umap_with_scores.pdf", bbox_inches="tight")
        plt.show()


def plot_entropy_boxplots(entropy_boxplot_data: pd.DataFrame, output_path: Path, palette: Dict[str, str]) -> None:
    """Plot boxplots for neighborhood entropy across different segmentation methods by cell type.

    Args:
    entropy_boxplot_data (pd.DataFrame): DataFrame containing neighborhood entropy data for all segmentation methods.
    output_path (Path): Path to the directory where the plot will be saved.
    palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.
    """
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        x="Cell Type", y="Neighborhood Entropy", hue="Segmentation Method", data=entropy_boxplot_data, palette=palette
    )
    plt.title("Neighborhood Entropy")
    plt.xlabel("Cell Type")
    plt.ylabel("Neighborhood Entropy")
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "neighborhood_entropy_boxplots.pdf", bbox_inches="tight")
    plt.show()


def plot_sensitivity_boxplots(
    sensitivity_boxplot_data: pd.DataFrame, output_path: Path, palette: Dict[str, str]
) -> None:
    """Plot boxplots for sensitivity across different segmentation methods by cell type.
    Args:
        sensitivity_boxplot_data (pd.DataFrame): DataFrame containing sensitivity data for all segmentation methods.
        output_path (Path): Path to the directory where the plot will be saved.
        palette (Dict[str, str]): Dictionary mapping segmentation method names to color codes.
    """
    sensitivity_boxplot_data.to_csv(output_path / "sensitivity_results.csv", index=True)
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        x="Cell Type", y="Sensitivity", hue="Segmentation Method", data=sensitivity_boxplot_data, palette=palette
    )
    plt.title("Sensitivity Score")
    plt.xlabel("Cell Type")
    plt.ylabel("Sensitivity")
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "sensitivity_boxplots.pdf", bbox_inches="tight")
    plt.show()
