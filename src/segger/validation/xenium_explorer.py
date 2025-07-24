import os
import sys
from pathlib import Path
import gzip
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from segger.prediction.boundary import generate_boundary
from shapely import Polygon
import zarr



def get_flatten_version(polygon_vertices: List[List[Tuple[float, float]]], max_value: int = 16) -> np.ndarray:
    """Standardize list of polygon vertices to a fixed shape.

    Args:
        polygon_vertices (List[List[Tuple[float, float]]]): List of polygon coordinate lists.
        max_value (int): Max number of coordinates per polygon.

    Returns:
        np.ndarray: Padded or truncated list of polygon vertices.
    """
    flattened = []
    for vertices in polygon_vertices:
        if isinstance(vertices, np.ndarray):
        # Handle numpy array case
            if len(vertices) >= max_value:
                return vertices[:max_value]
            padding = np.zeros((max_value - len(vertices), vertices.shape[1]))
            return np.concatenate([vertices, padding])
        if len(vertices) > max_value:
            flattened.append(vertices[:max_value])
        else:
            flattened.append(vertices + [(0.0, 0.0)] * (max_value - len(vertices)))
    return np.array(flattened, dtype=np.float32)


def seg2explorer(
    seg_df: pd.DataFrame,
    source_path: str,
    output_dir: str,
    cells_filename: str = "seg_cells",
    analysis_filename: str = "seg_analysis",
    xenium_filename: str = "seg_experiment.xenium",
    analysis_df: Optional[pd.DataFrame] = None,
    draw: bool = False,
    cell_id_columns: str = "seg_cell_id",
    area_low: float = 10,
    area_high: float = 100,
) -> None:
    """Convert segmentation results into a Xenium Explorer-compatible Zarr dataset.

    Args:
        seg_df (pd.DataFrame): Segmented transcript dataframe.
        source_path (str): Path to the original Zarr store.
        output_dir (str): Output directory to save new Zarr and Xenium files.
        cells_filename (str): Filename prefix for cell Zarr file.
        analysis_filename (str): Filename prefix for cell group Zarr file.
        xenium_filename (str): Output experiment filename for Xenium.
        analysis_df (Optional[pd.DataFrame]): Optional dataframe with cluster annotations.
        draw (bool): Whether to draw polygons (not used currently).
        cell_id_columns (str): Column containing cell IDs.
        area_low (float): Minimum area threshold to include cells.
        area_high (float): Maximum area threshold to include cells.
    """
    source_path = Path(source_path)
    storage = Path(output_dir)

    cell_id2old_id: Dict[int, Any] = {}
    cell_id: List[int] = []
    cell_summary: List[Dict[str, Any]] = []
    polygon_num_vertices: List[List[int]] = [[], []]
    polygon_vertices: List[List[Any]] = [[], []]
    seg_mask_value: List[int] = []

    grouped_by = seg_df.groupby(cell_id_columns)

    for cell_incremental_id, (seg_cell_id, seg_cell) in tqdm(
        enumerate(grouped_by), total=len(grouped_by)
    ):
        if len(seg_cell) < 5:
            continue

        cell_convex_hull = generate_boundary(seg_cell)
        if cell_convex_hull is None or not isinstance(cell_convex_hull, Polygon):
            continue

        if not (area_low <= cell_convex_hull.area <= area_high):
            continue

        uint_cell_id = cell_incremental_id + 1
        cell_id2old_id[uint_cell_id] = seg_cell_id

        seg_nucleous = seg_cell[seg_cell["overlaps_nucleus"] == 1]
        nucleus_convex_hull = None
        # if len(seg_nucleous) >= 3:
        #     try:
        #         nucleus_convex_hull = ConvexHull(seg_nucleous[["x_location", "y_location"]])
        #     except Exception:
        #         pass

        cell_id.append(uint_cell_id)
        cell_summary.append(
            {
                "cell_centroid_x": seg_cell["x_location"].mean(),
                "cell_centroid_y": seg_cell["y_location"].mean(),
                "cell_area": cell_convex_hull.area,
                "nucleus_centroid_x": seg_cell["x_location"].mean(),
                "nucleus_centroid_y": seg_cell["y_location"].mean(),
                "nucleus_area": cell_convex_hull.area,
                "z_level": (seg_cell.z_location.mean() // 3).round(0) * 3,
            }
        )
        polygon_num_vertices[0].append(len(cell_convex_hull.exterior.coords))
        # polygon_num_vertices[1].append(
        #     len(nucleus_convex_hull.vertices) if nucleus_convex_hull else 0
        # )
        polygon_vertices[0].append(list(cell_convex_hull.exterior.coords))
        # polygon_vertices[1].append(
        #     seg_nucleous[["x_location", "y_location"]].values[
        #         nucleus_convex_hull.vertices
        #     ]
        #     if nucleus_convex_hull else np.array([[], []]).T
        # )
        seg_mask_value.append(uint_cell_id)

    print(polygon_vertices[0][0])

    cell_polygon_vertices = get_flatten_version(polygon_vertices[0], max_value=21)
    # nucl_polygon_vertices = get_flatten_version(polygon_vertices[1], max_value=21)

    print(cell_polygon_vertices)
    print(cell_polygon_vertices.shape)
    # print(nucl_polygon_vertices)
    # print(nucl_polygon_vertices.shape)
    cells = {
        "cell_id": np.array(
            [np.array(cell_id), np.ones(len(cell_id))], dtype=np.uint32
        ).T,
        "cell_summary": pd.DataFrame(cell_summary).values.astype(np.float64),
        "polygon_num_vertices": np.array(
            [
                # [min(x + 1, x + 1) for x in polygon_num_vertices[1]],
                [min(x + 1, x + 1) for x in polygon_num_vertices[0]],
            ],
            dtype=np.int32,
        ),
        "polygon_vertices": np.array(
            [cell_polygon_vertices], dtype=np.float32
        ),
        "seg_mask_value": np.array(seg_mask_value, dtype=np.int32),
    }

    print(cells)

    existing_store = zarr.open(source_path / "cells.zarr.zip", mode="r")
    new_store = zarr.open(storage / f"{cells_filename}.zarr.zip", mode="w")
    new_store["cell_id"] = cells["cell_id"]
    new_store["polygon_num_vertices"] = cells["polygon_num_vertices"]
    new_store["polygon_vertices"] = cells["polygon_vertices"]
    new_store["seg_mask_value"] = cells["seg_mask_value"]
    new_store.attrs.update(existing_store.attrs)
    new_store.attrs["number_cells"] = len(cells["cell_id"])
    new_store.store.close()

    if analysis_df is None:
        analysis_df = pd.DataFrame(
            [cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns]
        )
        analysis_df["default"] = "seg"

    zarr_df = pd.DataFrame(
        [cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns]
    )
    clustering_df = pd.merge(zarr_df, analysis_df, how="left", on=cell_id_columns)
    clusters_names = [col for col in analysis_df.columns if col != cell_id_columns]

    clusters_dict = {
        cluster: {
            label: idx + 1
            for idx, label in enumerate(
                sorted(np.unique(clustering_df[cluster].dropna()))
            )
        }
        for cluster in clusters_names
    }

    new_zarr = zarr.open(storage / f"{analysis_filename}.zarr.zip", mode="w")
    new_zarr.create_group("/cell_groups")
    for i, cluster in enumerate(clusters_names):
        new_zarr["cell_groups"].create_group(str(i))
        group_values = [clusters_dict[cluster].get(x, 0) for x in clustering_df[cluster]]
        indices, indptr = get_indices_indptr(np.array(group_values))
        new_zarr["cell_groups"][str(i)]["indices"] = indices
        new_zarr["cell_groups"][str(i)]["indptr"] = indptr

    new_zarr["cell_groups"].attrs.update(
        {
            "major_version": 1,
            "minor_version": 0,
            "number_groupings": len(clusters_names),
            "grouping_names": clusters_names,
            "group_names": [
                sorted(clusters_dict[cluster], key=clusters_dict[cluster].get)
                for cluster in clusters_names
            ],
        }
    )
    new_zarr.store.close()

    generate_experiment_file(
        template_path=source_path / "experiment.xenium",
        output_path=storage / xenium_filename,
        cells_name=cells_filename,
        analysis_name=analysis_filename,
    )



def str_to_uint32(cell_id_str: str) -> Tuple[int, int]:
    """Convert a string cell ID back to uint32 format.

    Args:
        cell_id_str (str): The cell ID in string format.

    Returns:
        Tuple[int, int]: The cell ID in uint32 format and the dataset suffix.
    """
    prefix, suffix = cell_id_str.split("-")
    str_to_hex_mapping = {
        "a": "0",
        "b": "1",
        "c": "2",
        "d": "3",
        "e": "4",
        "f": "5",
        "g": "6",
        "h": "7",
        "i": "8",
        "j": "9",
        "k": "a",
        "l": "b",
        "m": "c",
        "n": "d",
        "o": "e",
        "p": "f",
    }
    hex_prefix = "".join([str_to_hex_mapping[char] for char in prefix])
    cell_id_uint32 = int(hex_prefix, 16)
    dataset_suffix = int(suffix)
    return cell_id_uint32, dataset_suffix


def get_indices_indptr(input_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the indices and indptr arrays for sparse matrix representation.

    Args:
        input_array (np.ndarray): The input array containing cluster labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The indices and indptr arrays.
    """
    clusters = sorted(np.unique(input_array[input_array != 0]))
    indptr = np.zeros(len(clusters), dtype=np.uint32)
    indices = []

    for cluster in clusters:
        cluster_indices = np.where(input_array == cluster)[0]
        indptr[cluster - 1] = len(indices)
        indices.extend(cluster_indices)

    indices.extend(-np.zeros(len(input_array[input_array == 0])))
    indices = np.array(indices, dtype=np.int32).astype(np.uint32)
    return indices, indptr


def save_cell_clustering(
    merged: pd.DataFrame, zarr_path: str, columns: List[str]
) -> None:
    """Save cell clustering information to a Zarr file.

    Args:
        merged (pd.DataFrame): The merged dataframe containing cell clustering information.
        zarr_path (str): The path to the Zarr file.
        columns (List[str]): The list of columns to save.
    """
    import zarr

    new_zarr = zarr.open(zarr_path, mode="w")
    new_zarr.create_group("/cell_groups")

    mappings = []
    for index, column in enumerate(columns):
        new_zarr["cell_groups"].create_group(index)
        classes = list(np.unique(merged[column].astype(str)))
        mapping_dict = {
            key: i
            for i, key in zip(
                range(1, len(classes)), [k for k in classes if k != "nan"]
            )
        }
        mapping_dict["nan"] = 0

        clusters = merged[column].astype(str).replace(mapping_dict).values.astype(int)
        indices, indptr = get_indices_indptr(clusters)

        new_zarr["cell_groups"][index].create_dataset("indices", data=indices)
        new_zarr["cell_groups"][index].create_dataset("indptr", data=indptr)
        mappings.append(mapping_dict)

    new_zarr["cell_groups"].attrs.update(
        {
            "major_version": 1,
            "minor_version": 0,
            "number_groupings": len(columns),
            "grouping_names": columns,
            "group_names": [
                [k for k, v in sorted(mapping_dict.items(), key=lambda item: item[1])][
                    1:
                ]
                for mapping_dict in mappings
            ],
        }
    )
    new_zarr.store.close()


def draw_umap(adata, column: str = "leiden") -> None:
    """Draw UMAP plots for the given AnnData object.

    Args:
        adata (AnnData): The AnnData object containing the data.
        column (str): The column to color the UMAP plot by.
    """
    sc.pl.umap(adata, color=[column])
    plt.show()

    sc.pl.umap(adata, color=["KRT5", "KRT7"], vmax="p95")
    plt.show()

    sc.pl.umap(adata, color=["ACTA2", "PTPRC"], vmax="p95")
    plt.show()


def get_leiden_umap(adata, draw: bool = False):
    """Perform Leiden clustering and UMAP visualization on the given AnnData object.

    Args:
        adata (AnnData): The AnnData object containing the data.
        draw (bool): Whether to draw the UMAP plots.

    Returns:
        AnnData: The AnnData object with Leiden clustering and UMAP results.
    """
    sc.pp.filter_cells(adata, min_genes=5)
    sc.pp.filter_genes(adata, min_cells=5)

    gene_names = adata.var_names
    mean_expression_values = adata.X.mean(axis=0)
    gene_mean_expression_df = pd.DataFrame(
        {"gene_name": gene_names, "mean_expression": mean_expression_values}
    )
    top_genes = gene_mean_expression_df.sort_values(
        by="mean_expression", ascending=False
    ).head(30)
    top_gene_names = top_genes["gene_name"].tolist()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

    if draw:
        draw_umap(adata, "leiden")

    return adata


def get_median_expression_table(adata, column: str = "leiden") -> pd.DataFrame:
    """Get the median expression table for the given AnnData object.

    Args:
        adata (AnnData): The AnnData object containing the data.
        column (str): The column to group by.

    Returns:
        pd.DataFrame: The median expression table.
    """
    top_genes = [
        "GATA3",
        "ACTA2",
        "KRT7",
        "KRT8",
        "KRT5",
        "AQP1",
        "SERPINA3",
        "PTGDS",
        "CXCR4",
        "SFRP1",
        "ENAH",
        "MYH11",
        "SVIL",
        "KRT14",
        "CD4",
    ]
    top_gene_indices = [adata.var_names.get_loc(gene) for gene in top_genes]

    clusters = adata.obs[column]
    cluster_data = {}

    for cluster in clusters.unique():
        cluster_cells = adata[clusters == cluster].X
        cluster_expression = cluster_cells[:, top_gene_indices]
        gene_medians = [
            pd.Series(cluster_expression[:, gene_idx]).median()
            for gene_idx in range(len(top_gene_indices))
        ]
        cluster_data[f"Cluster_{cluster}"] = gene_medians

    cluster_expression_df = pd.DataFrame(cluster_data, index=top_genes)
    sorted_columns = sorted(
        cluster_expression_df.columns.values, key=lambda x: int(x.split("_")[-1])
    )
    cluster_expression_df = cluster_expression_df[sorted_columns]
    return cluster_expression_df.T.style.background_gradient(cmap="Greens")


def generate_experiment_file(
    template_path: str,
    output_path: str,
    cells_name: str = "seg_cells",
    analysis_name: str = "seg_analysis",
) -> None:
    """Generate the experiment file for Xenium.

    Args:
        template_path (str): The path to the template file.
        output_path (str): The path to the output file.
        cells_name (str): The name of the cells file.
        analysis_name (str): The name of the analysis file.
    """
    import json

    with open(template_path) as f:
        experiment = json.load(f)

    # experiment["images"].pop("morphology_filepath")
    # experiment["images"].pop("morphology_focus_filepath")

    experiment["xenium_explorer_files"][
        "cells_zarr_filepath"
    ] = f"{cells_name}.zarr.zip"
    experiment["xenium_explorer_files"].pop("cell_features_zarr_filepath")
    experiment["xenium_explorer_files"][
        "analysis_zarr_filepath"
    ] = f"{analysis_name}.zarr.zip"

    with open(output_path, "w") as f:
        json.dump(experiment, f, indent=2)