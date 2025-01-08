from pqdm.processes import pqdm
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import zarr
from scipy.spatial import ConvexHull
from typing import Dict, Any, Optional, List, Tuple
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def get_boundary(seg_cell, x="x_location", y="y_location"):
    """Calculate the boundary for the cell."""
    if len(seg_cell) < 3:
        return None
    bi = BoundaryIdentification(seg_cell[[x, y]].values)
    bi.calculate_part_1(plot=False)
    bi.calculate_part_2(plot=False)
    return bi.find_cycles()


def process_group(group, area_low, area_high):
    """Process each group for parallel computation."""
    cell_incremental_id, (seg_cell_id, seg_cell) = group
    # print(seg_cell)
    # print(cell_incremental_id)
    # print(seg_cell)
    if len(seg_cell) < 5:
        return None
    # Cell boundary using `get_boundary`
    cell_boundary = generate_boundary(seg_cell)
    if isinstance(cell_boundary, MultiPolygon):
        # polygons = sorted(cell_boundary, key=lambda p: p.area, reverse=True)
        # Extract the largest polygon (by area) from MultiPolygon
        # cell_boundary = polygons[0]
        # cell_boundary = unary_union(cell_boundary)
        # polygons = sorted(cell_boundary, key=lambda p: p.area, reverse=True)
        # cell_boundary = polygons[0]
        cell_boundary = max(cell_boundary, key=lambda p: p.area)
    print(cell_boundary.area)
    # print(cell_boundary.area)
    if cell_boundary is None or cell_boundary.area > area_high or cell_boundary.area < area_low:
        # print('**********************************')
        # print(cell_boundary.area)
        # print('**********************************')
        return None
    uint_cell_id = cell_incremental_id + 1
    # print(uint_cell_id)
    # seg_nucleus = seg_cell[seg_cell["overlaps_nucleus"] == 1]
    # Nucleus boundary using ConvexHull
    # nucleus_boundary = None
    # if len(seg_nucleus) >= 3:
    #     try:
    #         nucleus_boundary = ConvexHull(seg_nucleus[["x_location", "y_location"]])
    #     except Exception:
    #         pass
    # Prepare data for the final output
    return {
        "uint_cell_id": uint_cell_id,
        "seg_cell_id": seg_cell_id,
        "cell_boundary": cell_boundary,
        # "nucleus_boundary": nucleus_boundary,
        "cell_summary": {
            "cell_centroid_x": seg_cell["x_location"].mean(),
            "cell_centroid_y": seg_cell["y_location"].mean(),
            "cell_area": cell_boundary.area,
            # "nucleus_centroid_x": seg_cell["x_location"].mean(),
            # "nucleus_centroid_y": seg_cell["y_location"].mean(),
            # "nucleus_area": nucleus_boundary.area if nucleus_boundary else 0,
            "z_level": (seg_cell.z_location.mean() // 3).round(0) * 3,
        },
        "seg_mask_value": uint_cell_id,
    }


def get_coordinates(boundary):
    """Extracts coordinates from a Polygon or MultiPolygon."""
    if isinstance(boundary, MultiPolygon):
        # Combine coordinates from all polygons in the MultiPolygon
        coords = []
        for polygon in boundary.geoms:
            coords.extend(polygon.exterior.coords)
        return coords
    elif isinstance(boundary, Polygon):
        # Return coordinates from a single Polygon
        return list(boundary.exterior.coords)


def get_flatten_version(polygon_vertices, max_value=21):
    """
    Flattens and standardizes the shape of polygon vertices to a fixed length.

    Args:
        polygon_vertices (list): List of polygons where each polygon is a list of coordinates.
        max_value (int): The fixed number of vertices per polygon.

    Returns:
        np.array: A standardized array of polygons with exactly max_value vertices each.
    """
    flattened = []
    for vertices in polygon_vertices:
        # Pad or truncate each polygon to the max_value
        if len(vertices) > max_value:
            flattened.append(vertices[:max_value])
        else:
            flattened.append(vertices + [(0.0, 0.0)] * (max_value - len(vertices)))
    return np.array(flattened, dtype=np.float32)


from segger.validation.xenium_explorer import *
from segger.prediction.boundary import *


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
    n_jobs: int = 4,
) -> None:
    """Convert seg output to a format compatible with Xenium explorer."""
    source_path = Path(source_path)
    storage = Path(output_dir)
    # Group by cell_id_columns
    grouped_by = list(seg_df.groupby(cell_id_columns))
    # Process groups in parallel with pqdm
    # results = pqdm(
    #     tqdm(enumerate(grouped_by), desc="Processing Cells", total=len(grouped_by)),
    #     lambda group: process_group(group, area_low, area_high),
    #     n_jobs=n_jobs,
    # )
    results = []
    for idx, group in tqdm(enumerate(grouped_by), desc="Processing Cells", total=len(grouped_by)):
        try:
            res = process_group((idx, group), area_low, area_high)
            if isinstance(res, dict):  # Only keep valid results
                results.append(res)
            else:
                print(f"Invalid result at index {idx}: {res}")
        except Exception as e:
            print(f"Error processing group at index {idx}: {e}")
    # print(results)
    print(results[0])
    # Filter out None results
    results = [res for res in results if res]
    # print('********************************0')
    # Extract processed data
    cell_id2old_id = {res["uint_cell_id"]: res["seg_cell_id"] for res in results}
    cell_id = [res["uint_cell_id"] for res in results]
    cell_summary = [res["cell_summary"] for res in results]
    # print('********************************1')
    polygon_num_vertices = [len(res["cell_boundary"].exterior.coords) for res in results]
    print(polygon_num_vertices)
    # polygon_vertices = np.array(
    #     [list(res["cell_boundary"].exterior.coords) for res in results],
    #     dtype=object
    # )
    polygon_vertices = np.array([list(res["cell_boundary"].exterior.coords) for res in results], dtype=object)
    # polygon_vertices = get_flatten_version(
    #     [get_coordinates(res["cell_boundary"]) for res in results],
    #     max_value=21,
    # )
    # print(polygon_vertices)
    # print('********************************2')
    seg_mask_value = [res["seg_mask_value"] for res in results]
    # Convert results to Zarr
    cells = {
        "cell_id": np.array([np.array(cell_id), np.ones(len(cell_id))], dtype=np.uint32).T,
        "cell_summary": pd.DataFrame(cell_summary).values.astype(np.float64),
        "polygon_num_vertices": np.array(polygon_num_vertices).astype(np.int32),
        "polygon_vertices": np.array(polygon_vertices).astype(np.float32),
        "seg_mask_value": np.array(seg_mask_value, dtype=np.int32),
    }
    print(len(cells["cell_id"]))
    # Save cells data
    existing_store = zarr.open(source_path / "cells.zarr.zip", mode="r")
    new_store = zarr.open(storage / f"{cells_filename}.zarr.zip", mode="w")
    new_store["cell_id"] = cells["cell_id"]
    new_store["polygon_num_vertices"] = cells["polygon_num_vertices"]
    new_store["polygon_vertices"] = cells["polygon_vertices"]
    new_store["seg_mask_value"] = cells["seg_mask_value"]
    new_store.attrs.update(existing_store.attrs)
    new_store.attrs["number_cells"] = len(cells["cell_id"])
    print(new_store)
    new_store.store.close()

    print(cells["polygon_vertices"])
    # # Save analysis data
    if analysis_df is None:
        analysis_df = pd.DataFrame([cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns])
        analysis_df["default"] = "seg"
    zarr_df = pd.DataFrame([cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns])
    clustering_df = pd.merge(zarr_df, analysis_df, how="left", on=cell_id_columns)
    clusters_names = [i for i in analysis_df.columns if i != cell_id_columns]
    clusters_dict = {
        cluster: {
            j: i
            for i, j in zip(
                range(1, len(sorted(np.unique(clustering_df[cluster].dropna()))) + 1),
                sorted(np.unique(clustering_df[cluster].dropna())),
            )
        }
        for cluster in clusters_names
    }
    new_zarr = zarr.open(storage / (analysis_filename + ".zarr.zip"), mode="w")
    new_zarr.create_group("/cell_groups")
    clusters = [[clusters_dict[cluster].get(x, 0) for x in list(clustering_df[cluster])] for cluster in clusters_names]
    for i in range(len(clusters)):
        new_zarr["cell_groups"].create_group(i)
        indices, indptr = get_indices_indptr(np.array(clusters[i]))
        new_zarr["cell_groups"][i].create_dataset("indices", data=indices)
        new_zarr["cell_groups"][i].create_dataset("indptr", data=indptr)
    new_zarr["cell_groups"].attrs.update(
        {
            "major_version": 1,
            "minor_version": 0,
            "number_groupings": len(clusters_names),
            "grouping_names": clusters_names,
            "group_names": [
                [x[0] for x in sorted(clusters_dict[cluster].items(), key=lambda x: x[1])] for cluster in clusters_names
            ],
        }
    )
    new_zarr.store.close()
    # Generate experiment file
    generate_experiment_file(
        template_path=source_path / "experiment.xenium",
        output_path=storage / xenium_filename,
        cells_name=cells_filename,
        analysis_name=analysis_filename,
    )


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
    """Convert seg output to a format compatible with Xenium explorer.

    Args:
        seg_df (pd.DataFrame): The seg DataFrame.
        source_path (str): The source path.
        output_dir (str): The output directory.
        cells_filename (str): The filename for cells.
        analysis_filename (str): The filename for analysis.
        xenium_filename (str): The filename for Xenium.
        analysis_df (Optional[pd.DataFrame]): The analysis DataFrame.
        draw (bool): Whether to draw the plots.
        cell_id_columns (str): The cell ID columns.
        area_low (float): The lower area threshold.
        area_high (float): The upper area threshold.
    """
    import zarr
    import json

    source_path = Path(source_path)
    storage = Path(output_dir)
    cell_id2old_id = {}
    cell_id = []
    cell_summary = []
    polygon_num_vertices = [[], []]
    polygon_vertices = [[], []]
    seg_mask_value = []
    tma_id = []
    grouped_by = seg_df.groupby(cell_id_columns)
    for cell_incremental_id, (seg_cell_id, seg_cell) in tqdm(enumerate(grouped_by), total=len(grouped_by)):
        if len(seg_cell) < 5:
            continue
        # print('****************1')
        # cell_convex_hull = ConvexHull(seg_cell[["x_location", "y_location"]])
        # print('****************2')
        # hull_vertices = [seg_cell[["x_location", "y_location"]].values[vertex] for vertex in cell_convex_hull.vertices]
        # print('****************3')
        # # Create a Shapely Polygon
        # cell_convex_hull  = Polygon(hull_vertices)
        cell_convex_hull = get_boundary(seg_cell)
        # print(cell_convex_hull)
        if isinstance(cell_convex_hull, MultiPolygon):
            # polygons = sorted(cell_boundary, key=lambda p: p.area, reverse=True)
            # Extract the largest polygon (by area) from MultiPolygon
            # cell_boundary = polygons[0]
            # cell_convex_hull = unary_union(cell_convex_hull)
            # print('****************1')
            # polygons = sorted(cell_convex_hull.geoms, key=lambda p: p.area, reverse=True)
            # cell_convex_hull = polygons[0]
            continue
            # print('****************2')
            # cell_convex_hull = max(cell_convex_hull.geoms, key=lambda p: p.area)
        # print(cell_convex_hull)
        if cell_convex_hull.area > area_high:
            continue
        if cell_convex_hull.area < area_low:
            continue
        uint_cell_id = cell_incremental_id + 1
        cell_id2old_id[uint_cell_id] = seg_cell_id
        seg_nucleous = seg_cell[seg_cell["overlaps_nucleus"] == 1]
        if len(seg_nucleous) >= 3:
            nucleus_convex_hull = ConvexHull(seg_nucleous[["x_location", "y_location"]])
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
        polygon_num_vertices[1].append(len(nucleus_convex_hull.vertices) if len(seg_nucleous) >= 3 else 0)
        polygon_vertices[0].append(cell_convex_hull.exterior.coords)
        polygon_vertices[1].append(
            seg_nucleous[["x_location", "y_location"]].values[nucleus_convex_hull.vertices]
            if len(seg_nucleous) >= 3
            else np.array([[], []]).T
        )
        seg_mask_value.append(cell_incremental_id + 1)
    cell_polygon_vertices = get_flatten_version(polygon_vertices[0], max_value=128)
    nucl_polygon_vertices = get_flatten_version(polygon_vertices[1], max_value=128)
    cells = {
        "cell_id": np.array([np.array(cell_id), np.ones(len(cell_id))], dtype=np.uint32).T,
        "cell_summary": pd.DataFrame(cell_summary).values.astype(np.float64),
        "polygon_num_vertices": np.array(
            [
                [min(x + 1, x + 1) for x in polygon_num_vertices[1]],
                [min(x + 1, x + 1) for x in polygon_num_vertices[0]],
            ],
            dtype=np.int32,
        ),
        "polygon_vertices": np.array([nucl_polygon_vertices, cell_polygon_vertices]).astype(np.float32),
        "seg_mask_value": np.array(seg_mask_value, dtype=np.int32),
    }
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
        analysis_df = pd.DataFrame([cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns])
        analysis_df["default"] = "seg"
    zarr_df = pd.DataFrame([cell_id2old_id[i] for i in cell_id], columns=[cell_id_columns])
    clustering_df = pd.merge(zarr_df, analysis_df, how="left", on=cell_id_columns)
    clusters_names = [i for i in analysis_df.columns if i != cell_id_columns]
    clusters_dict = {
        cluster: {
            j: i
            for i, j in zip(
                range(1, len(sorted(np.unique(clustering_df[cluster].dropna()))) + 1),
                sorted(np.unique(clustering_df[cluster].dropna())),
            )
        }
        for cluster in clusters_names
    }
    new_zarr = zarr.open(storage / (analysis_filename + ".zarr.zip"), mode="w")
    new_zarr.create_group("/cell_groups")
    clusters = [[clusters_dict[cluster].get(x, 0) for x in list(clustering_df[cluster])] for cluster in clusters_names]
    for i in range(len(clusters)):
        new_zarr["cell_groups"].create_group(i)
        indices, indptr = get_indices_indptr(np.array(clusters[i]))
        new_zarr["cell_groups"][i].create_dataset("indices", data=indices)
        new_zarr["cell_groups"][i].create_dataset("indptr", data=indptr)
    new_zarr["cell_groups"].attrs.update(
        {
            "major_version": 1,
            "minor_version": 0,
            "number_groupings": len(clusters_names),
            "grouping_names": clusters_names,
            "group_names": [
                [x[0] for x in sorted(clusters_dict[cluster].items(), key=lambda x: x[1])] for cluster in clusters_names
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


ddf = dd.read_parquet(
    "/dkfz/cluster/gpu/data/OE0606/elihei/segger_experiments/data_tidy/benchmarks/xe_rep1_bc/parquet_train_big_0.5_False_3_10_5_3_20241030/segger_transcripts.parquet"
).compute()
ddf = ddf.dropna()
ddf = ddf[ddf.segger_cell_id != "None"]
ddf = ddf.sort_values("segger_cell_id")
df = ddf.iloc[:10000, :]


df_path = Path("data_tidy/Xenium_FFPE_Human_Breast_Cancer_Rep1_v9_segger.csv.gz")
df_v9 = dd.read_csv(df_path)
df_main = dd.read_parquet("data_raw/breast_cancer/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs/transcripts.parquet")

ddf = df_v9.merge(df_main, on="transcript_id")
ddf = ddf.compute()
ddf = ddf[ddf.segger_cell_id != "None"]
ddf = ddf.sort_values("segger_cell_id")
df = ddf.loc[(ddf.x_location > 250) & (ddf.x_location < 1500) & (ddf.y_location > 500) & (ddf.y_location < 1500), :]

# tx_df = dd.read_csv('data_tidy/Xenium_FFPE_Human_Breast_Cancer_Rep2_v9_segger.csv.gz')
# ddf = tx_df.merge(df_main, on='transcript_id')

seg2explorer(
    seg_df=df,
    source_path="data_raw/breast_cancer/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs",
    output_dir="data_tidy/explorer/rep1sis",
    cells_filename="segger_cells_seg_roi1",
    analysis_filename="segger_analysis_seg_roi1",
    xenium_filename="segger_experiment_seg_roi1.xenium",
    analysis_df=None,
    cell_id_columns="segger_cell_id",
    area_low=10,
    area_high=1000,
)


df = ddf.loc[(ddf.x_location > 1550) & (ddf.x_location < 3250) & (ddf.y_location > 2250) & (ddf.y_location < 3550), :]

# tx_df = dd.read_csv('data_tidy/Xenium_FFPE_Human_Breast_Cancer_Rep2_v9_segger.csv.gz')
# ddf = tx_df.merge(df_main, on='transcript_id')

seg2explorer(
    seg_df=df,
    source_path="data_raw/breast_cancer/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs",
    output_dir="data_tidy/explorer/rep1sis",
    cells_filename="segger_cells_seg_roi2",
    analysis_filename="segger_analysis_seg_roi2",
    xenium_filename="segger_experiment_seg_roi2.xenium",
    analysis_df=None,
    cell_id_columns="segger_cell_id",
    area_low=10,
    area_high=1000,
)


df = ddf.loc[(ddf.x_location > 4000) & (ddf.x_location < 4500) & (ddf.y_location > 1000) & (ddf.y_location < 1500), :]

# tx_df = dd.read_csv('data_tidy/Xenium_FFPE_Human_Breast_Cancer_Rep2_v9_segger.csv.gz')
# ddf = tx_df.merge(df_main, on='transcript_id')

seg2explorer(
    seg_df=df,
    source_path="data_raw/breast_cancer/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs",
    output_dir="data_tidy/explorer/rep1sis",
    cells_filename="segger_cells_seg_roi3",
    analysis_filename="segger_analysis_seg_roi3",
    xenium_filename="segger_experiment_seg_roi3.xenium",
    analysis_df=None,
    cell_id_columns="segger_cell_id",
    area_low=10,
    area_high=1000,
)


df = ddf.loc[(ddf.x_location > 1550) & (ddf.x_location < 3250) & (ddf.y_location > 2250) & (ddf.y_location < 3550), :]

# tx_df = dd.read_csv('data_tidy/Xenium_FFPE_Human_Breast_Cancer_Rep2_v9_segger.csv.gz')
# ddf = tx_df.merge(df_main, on='transcript_id')

seg2explorer(
    seg_df=df,
    source_path="data_raw/breast_cancer/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs",
    output_dir="data_tidy/explorer/rep1sis",
    cells_filename="segger_cells_seg_roi2",
    analysis_filename="segger_analysis_seg_roi2",
    xenium_filename="segger_experiment_seg_roi2.xenium",
    analysis_df=None,
    cell_id_columns="segger_cell_id",
    area_low=10,
    area_high=1000,
)
