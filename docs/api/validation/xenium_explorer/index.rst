segger.validation.xenium_explorer
=================================

.. py:module:: segger.validation.xenium_explorer




Module Contents
---------------

.. py:function:: uint32_to_str(cell_id_uint32: int, dataset_suffix: int) -> str

   Convert a uint32 cell ID to a string format.

   Parameters:
   cell_id_uint32 (int): The cell ID in uint32 format.
   dataset_suffix (int): The dataset suffix to append to the cell ID.

   Returns:
   str: The cell ID in string format.


.. py:function:: str_to_uint32(cell_id_str: str) -> Tuple[int, int]

   Convert a string cell ID back to uint32 format.

   Parameters:
   cell_id_str (str): The cell ID in string format.

   Returns:
   Tuple[int, int]: The cell ID in uint32 format and the dataset suffix.


.. py:function:: get_indices_indptr(input_array: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]

   Get the indices and indptr arrays for sparse matrix representation.

   Parameters:
   input_array (np.ndarray): The input array containing cluster labels.

   Returns:
   Tuple[np.ndarray, np.ndarray]: The indices and indptr arrays.


.. py:function:: save_cell_clustering(merged: pandas.DataFrame, zarr_path: str, columns: List[str]) -> None

   Save cell clustering information to a Zarr file.

   Parameters:
   merged (pd.DataFrame): The merged dataframe containing cell clustering information.
   zarr_path (str): The path to the Zarr file.
   columns (List[str]): The list of columns to save.


.. py:function:: draw_umap(adata, column: str = 'leiden') -> None

   Draw UMAP plots for the given AnnData object.

   Parameters:
   adata (AnnData): The AnnData object containing the data.
   column (str): The column to color the UMAP plot by.


.. py:function:: get_leiden_umap(adata, draw: bool = False)

   Perform Leiden clustering and UMAP visualization on the given AnnData object.

   Parameters:
   adata (AnnData): The AnnData object containing the data.
   draw (bool): Whether to draw the UMAP plots.

   Returns:
   AnnData: The AnnData object with Leiden clustering and UMAP results.


.. py:function:: get_median_expression_table(adata, column: str = 'leiden') -> pandas.DataFrame

   Get the median expression table for the given AnnData object.

   Parameters:
   adata (AnnData): The AnnData object containing the data.
   column (str): The column to group by.

   Returns:
   pd.DataFrame: The median expression table.


.. py:function:: seg2explorer(seg_df: pandas.DataFrame, source_path: str, output_dir: str, cells_filename: str = 'seg_cells', analysis_filename: str = 'seg_analysis', xenium_filename: str = 'seg_experiment.xenium', analysis_df: Optional[pandas.DataFrame] = None, draw: bool = False, cell_id_columns: str = 'seg_cell_id', area_low: float = 10, area_high: float = 100) -> None

   Convert seg output to a format compatible with Xenium explorer.

   Parameters:
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


.. py:function:: get_flatten_version(polygons: List[numpy.ndarray], max_value: int = 21) -> numpy.ndarray

   Get the flattened version of polygon vertices.

   Parameters:
   polygons (List[np.ndarray]): List of polygon vertices.
   max_value (int): The maximum number of vertices to keep.

   Returns:
   np.ndarray: The flattened array of polygon vertices.


.. py:function:: generate_experiment_file(template_path: str, output_path: str, cells_name: str = 'seg_cells', analysis_name: str = 'seg_analysis') -> None

   Generate the experiment file for Xenium.

   Parameters:
   template_path (str): The path to the template file.
   output_path (str): The path to the output file.
   cells_name (str): The name of the cells file.
   analysis_name (str): The name of the analysis file.


