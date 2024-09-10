segger.data.io
==============

.. py:module:: segger.data.io




Module Contents
---------------

.. py:class:: SpatialTranscriptomicsSample(transcripts_df: pandas.DataFrame = None, transcripts_radius: int = 10, boundaries_graph: bool = False, keys: Dict = None)

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: transcripts_df


   .. py:attribute:: transcripts_radius


   .. py:attribute:: boundaries_graph


   .. py:attribute:: keys


   .. py:attribute:: embedding_df
      :value: None



   .. py:attribute:: current_embedding
      :value: 'one_hot'



   .. py:method:: _set_bounds() -> None

      Set the bounding box limits based on the current transcripts dataframe using lazy evaluation with Dask.



   .. py:method:: filter_transcripts(transcripts_df: pandas.DataFrame, min_qv: float = 20.0) -> pandas.DataFrame
      :abstractmethod:


      Abstract method to filter transcripts based on dataset-specific criteria.

      Parameters:
      -----------
      transcripts_df : pd.DataFrame
          The dataframe containing transcript data.
      min_qv : float, optional
          The minimum quality value threshold for filtering transcripts.

      Returns:
      --------
      :
      pd.DataFrame
          The filtered dataframe.



   .. py:method:: set_file_paths(transcripts_path: pathlib.Path, boundaries_path: pathlib.Path) -> None

      Set the paths for the transcript and boundary files.

      Parameters:
      -----------
      transcripts_path : Path
          Path to the Parquet file containing transcripts data.
      boundaries_path : Path
          Path to the Parquet file containing boundaries data.



   .. py:method:: load_transcripts(base_path: pathlib.Path = None, sample: str = None, transcripts_filename: str = None, path: pathlib.Path = None, file_format: str = 'parquet', x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None) -> dask.dataframe.DataFrame

      Load transcripts from a Parquet file using Dask for efficient chunked processing,
      only within the specified bounding box, and return the filtered DataFrame with integer token embeddings.

      Parameters:
      -----------
      base_path : Path, optional
          The base directory path where samples are stored.
      sample : str, optional
          The sample name or identifier.
      transcripts_filename : str, optional
          The filename of the transcripts file (default is derived from the dataset keys).
      path : Path, optional
          Specific path to the transcripts file.
      file_format : str, optional
          Format of the file to load (default is 'parquet').
      x_min : float, optional
          Minimum X-coordinate for the bounding box.
      x_max : float, optional
          Maximum X-coordinate for the bounding box.
      y_min : float, optional
          Minimum Y-coordinate for the bounding box.
      y_max : float, optional
          Maximum Y-coordinate for the bounding box.
      additional_embeddings : Dict[str, pd.DataFrame], optional
          A dictionary of additional embeddings for genes.

      Returns:
      --------
      :
      dd.DataFrame
          The filtered transcripts DataFrame.



   .. py:method:: load_boundaries(path: pathlib.Path, file_format: str = 'parquet', x_min: float = None, x_max: float = None, y_min: float = None, y_max: float = None) -> dask.dataframe.DataFrame

      Load boundaries data lazily using Dask, filtering by the specified bounding box.

      Parameters:
      -----------
      path : Path
          Path to the boundaries file.
      file_format : str, optional
          Format of the file to load. Only 'parquet' is supported in this refactor.
      x_min : float, optional
          Minimum X-coordinate for the bounding box.
      x_max : float, optional
          Maximum X-coordinate for the bounding box.
      y_min : float, optional
          Minimum Y-coordinate for the bounding box.
      y_max : float, optional
          Maximum Y-coordinate for the bounding box.

      Returns:
      --------
      :
      dd.DataFrame
          The filtered boundaries DataFrame.



   .. py:method:: set_metadata() -> None

      Set metadata for the transcript dataset, including bounding box limits and unique gene names,
      without reading the entire Parquet file. Additionally, return integer tokens for unique gene names
      instead of one-hot encodings and store the lookup table for later mapping.



   .. py:method:: set_embedding(embedding_name: str) -> None

      Set the current embedding type for the transcripts.

      Parameters:
      -----------
      embedding_name : str
          The name of the embedding to use.

      Returns:
      --------
      :
      None



   .. py:method:: get_tile_data(x_min: float, y_min: float, x_size: float, y_size: float) -> Tuple[dask.dataframe.DataFrame, dask.dataframe.DataFrame]

      Load the necessary data for a given tile from the transcripts and boundaries.
      Uses Dask's filtering for chunked processing.
      Parameters:
      -----------
      x_min : float
          Minimum x-coordinate of the tile.
      y_min : float
          Minimum y-coordinate of the tile.
      x_size : float
          Width of the tile.
      y_size : float
          Height of the tile.
      Returns:
      --------
      :
      Tuple[dd.DataFrame, dd.DataFrame]
          Transcripts and boundaries data for the tile.



   .. py:method:: get_bounding_box(x_min: float = None, y_min: float = None, x_max: float = None, y_max: float = None, in_place: bool = True) -> Optional[SpatialTranscriptomicsSample]

      Subsets the transcripts_df and boundaries_df within the specified bounding box using Dask.
      Parameters:
      -----------
      x_min : float, optional
          The minimum x-coordinate of the bounding box.
      y_min : float, optional
          The minimum y-coordinate of the bounding box.
      x_max : float, optional
          The maximum x-coordinate of the bounding box.
      y_max : float, optional
          The maximum y-coordinate of the bounding box.
      in_place : bool, optional
          If True, modifies the current instance. If False, returns a new instance with the subsetted data.
      Returns:
      --------
      :
      Optional[SpatialTranscriptomicsSample]
          If in_place is True, returns None after modifying the existing instance.
          If in_place is False, returns a new SpatialTranscriptomicsSample instance with the subsetted data.



   .. py:method:: create_scaled_polygon(group: pandas.DataFrame, scale_factor: float, keys) -> geopandas.GeoDataFrame
      :staticmethod:


      Static method to create and scale a polygon from boundary vertices and return a GeoDataFrame.

      This method ensures that the polygon scaling happens efficiently without requiring
      an instance of the class. Keys should be passed to avoid Dask tokenization issues.

      Parameters:
      -----------
      group : pd.DataFrame
          Group of boundary coordinates (for a specific cell).
      scale_factor : float
          The factor by which to scale the polygons.
      keys : Enum or dict-like
          A collection of keys to access column names for 'cell_id', 'vertex_x', and 'vertex_y'.

      Returns:
      --------
      :
      gpd.GeoDataFrame
          A GeoDataFrame containing the scaled Polygon and cell_id.



   .. py:method:: generate_and_scale_polygons(boundaries_df: dask.dataframe.DataFrame, scale_factor: float = 1.0) -> dask_geopandas.GeoDataFrame

      Generate and scale polygons from boundary coordinates using Dask.
      Keeps class structure intact by using static method for the core polygon generation.

      Parameters:
      -----------
      boundaries_df : dask.dataframe.DataFrame
          DataFrame containing boundary coordinates.
      scale_factor : float, optional
          The factor by which to scale the polygons (default is 1.0).

      Returns:
      --------
      :
      dask_geopandas.GeoDataFrame
          A GeoDataFrame containing scaled Polygon objects and their centroids.



   .. py:method:: compute_transcript_overlap_with_boundaries(transcripts_df: dask.dataframe.DataFrame, boundaries_df: dask.dataframe.DataFrame = None, polygons_gdf: dask_geopandas.GeoDataFrame = None, scale_factor: float = 1.0) -> dask.dataframe.DataFrame

      Computes the overlap of transcript locations with scaled boundary polygons
      and assigns corresponding cell IDs to the transcripts using Dask.

      Parameters:
      -----------
      transcripts_df : dask.dataframe.DataFrame
          Dask DataFrame containing transcript data.
      boundaries_df : dask.dataframe.DataFrame, optional
          Dask DataFrame containing boundary data. Required if polygons_gdf is not provided.
      polygons_gdf : dask_geopandas.GeoDataFrame, optional
          Precomputed Dask GeoDataFrame containing boundary polygons. If None, will compute from boundaries_df.
      scale_factor : float, optional
          The factor by which to scale the boundary polygons. Default is 1.0.

      Returns:
      --------
      :
      dask.dataframe.DataFrame
          The updated DataFrame with overlap information (True for overlap, False for no overlap)
          and assigned cell IDs.



   .. py:method:: compute_boundaries_geometries(boundaries_df: dask.dataframe.DataFrame = None, polygons_gdf: dask_geopandas.GeoDataFrame = None, scale_factor: float = 1.0, area: bool = True, convexity: bool = True, elongation: bool = True, circularity: bool = True) -> dask_geopandas.GeoDataFrame

      Computes geometries for boundaries (e.g., nuclei, cells) from the dataframe using Dask.

      Parameters:
      -----------
      boundaries_df : dask.dataframe.DataFrame, optional
          The dataframe containing boundaries data. Required if polygons_gdf is not provided.
      polygons_gdf : dask_geopandas.GeoDataFrame, optional
          Precomputed Dask GeoDataFrame containing boundary polygons. If None, will compute from boundaries_df.
      scale_factor : float, optional
          The factor by which to scale the polygons (default is 1.0, no scaling).
      area : bool, optional
          Whether to compute area.
      convexity : bool, optional
          Whether to compute convexity.
      elongation : bool, optional
          Whether to compute elongation.
      circularity : bool, optional
          Whether to compute circularity.

      Returns:
      --------
      :
      dask_geopandas.GeoDataFrame
          A GeoDataFrame containing computed geometries.



   .. py:method:: _validate_bbox(x_min: float, y_min: float, x_max: float, y_max: float) -> Tuple[float, float, float, float]

      Validates and sets default values for bounding box coordinates.



   .. py:method:: save_dataset_for_segger(processed_dir: pathlib.Path, x_size: float = 1000, y_size: float = 1000, d_x: float = 900, d_y: float = 900, margin_x: float = None, margin_y: float = None, compute_labels: bool = True, r_tx: float = 5, k_tx: int = 3, val_prob: float = 0.1, test_prob: float = 0.2, neg_sampling_ratio_approx: float = 5, sampling_rate: float = 1, num_workers: int = 1, receptive_field: Dict[str, float] = {'k_bd': 4, 'dist_bd': 20, 'k_tx': 5, 'dist_tx': 10}, method: str = 'kd_tree', gpu: bool = False, workers: int = 1) -> None

      Saves the dataset for Segger in a processed format using Dask for parallel and lazy processing.

      Parameters:
      -----------
      processed_dir : Path
          Directory to save the processed dataset.
      x_size : float, optional
          Width of each tile.
      y_size : float, optional
          Height of each tile.
      d_x : float, optional
          Step size in the x direction for tiles.
      d_y : float, optional
          Step size in the y direction for tiles.
      margin_x : float, optional
          Margin in the x direction to include transcripts.
      margin_y : float, optional
          Margin in the y direction to include transcripts.
      compute_labels : bool, optional
          Whether to compute edge labels for tx_belongs_bd edges.
      r_tx : float, optional
          Radius for building the transcript-to-transcript graph.
      k_tx : int, optional
          Number of nearest neighbors for the tx-tx graph.
      val_prob : float, optional
          Probability of assigning a tile to the validation set.
      test_prob : float, optional
          Probability of assigning a tile to the test set.
      neg_sampling_ratio_approx : float, optional
          Approximate ratio of negative samples.
      sampling_rate : float, optional
          Rate of sampling tiles.
      num_workers : int, optional
          Number of workers to use for parallel processing.
      receptive_field : dict, optional
          Dictionary containing the values for 'k_bd', 'dist_bd', 'k_tx', and 'dist_tx'.
      method : str, optional
          Method for computing edge indices (e.g., 'kd_tree', 'faiss').
      gpu : bool, optional
          Whether to use GPU acceleration for edge index computation.
      workers : int, optional
          Number of workers to use to compute the neighborhood graph (per tile).

      Returns:
      --------
      :
      None



   .. py:method:: _prepare_directories(processed_dir: pathlib.Path) -> None

      Prepares directories for saving tiles.



   .. py:method:: _get_ranges(d_x: float, d_y: float) -> Tuple[numpy.ndarray, numpy.ndarray]

      Generates ranges for tiling.



   .. py:method:: _generate_tile_params(x_range: numpy.ndarray, y_range: numpy.ndarray, x_size: float, y_size: float, margin_x: float, margin_y: float, compute_labels: bool, r_tx: float, k_tx: int, val_prob: float, test_prob: float, neg_sampling_ratio_approx: float, sampling_rate: float, processed_dir: pathlib.Path, receptive_field: Dict[str, float], method: str, gpu: bool, workers: int) -> List[Tuple]

      Generates parameters for processing tiles using the bounding box approach.
      This version eliminates masks and directly uses the tile ranges and margins.

      Parameters are the same as the previous version.



   .. py:method:: _process_tile(tile_params: Tuple) -> None

      Process a single tile using Dask for parallelism and lazy evaluation, and save the data.

      Parameters:
      -----------
      tile_params : tuple
          Parameters for the tile processing.

      Returns:
      --------
      :
      None



   .. py:method:: build_pyg_data_from_tile(boundaries_df: dask.dataframe.DataFrame, transcripts_df: dask.dataframe.DataFrame, r_tx: float = 5.0, k_tx: int = 3, method: str = 'kd_tree', gpu: bool = False, workers: int = 1) -> torch_geometric.data.HeteroData

      Builds PyG data from a tile of boundaries and transcripts data using Dask utilities for efficient processing.

      Parameters:
      -----------
      boundaries_df : dd.DataFrame
          Dask DataFrame containing boundaries data (e.g., nucleus, cell).
      transcripts_df : dd.DataFrame
          Dask DataFrame containing transcripts data.
      r_tx : float
          Radius for building the transcript-to-transcript graph.
      k_tx : int
          Number of nearest neighbors for the tx-tx graph.
      method : str, optional
          Method for computing edge indices (e.g., 'kd_tree', 'faiss').
      gpu : bool, optional
          Whether to use GPU acceleration for edge index computation.
      workers : int, optional
          Number of workers to use for parallel processing.

      Returns:
      --------
      :
      HeteroData
          PyG Heterogeneous Data object.



.. py:class:: XeniumSample(transcripts_df: dask.dataframe.DataFrame = None, transcripts_radius: int = 10, boundaries_graph: bool = False)

   Bases: :py:obj:`SpatialTranscriptomicsSample`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: filter_transcripts(transcripts_df: dask.dataframe.DataFrame, min_qv: float = 20.0) -> dask.dataframe.DataFrame

      Filters transcripts based on quality value and removes unwanted transcripts for Xenium using Dask.

      Parameters:
      -----------
      transcripts_df : dd.DataFrame
          The Dask DataFrame containing transcript data.
      min_qv : float, optional
          The minimum quality value threshold for filtering transcripts.

      Returns:
      --------
      :
      dd.DataFrame
          The filtered Dask DataFrame.



.. py:class:: MerscopeSample(transcripts_df: dask.dataframe.DataFrame = None, transcripts_radius: int = 10, boundaries_graph: bool = False)

   Bases: :py:obj:`SpatialTranscriptomicsSample`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: filter_transcripts(transcripts_df: dask.dataframe.DataFrame, min_qv: float = 20.0) -> dask.dataframe.DataFrame

      Filters transcripts based on specific criteria for Merscope using Dask.

      Parameters:
      -----------
      transcripts_df : dd.DataFrame
          The Dask DataFrame containing transcript data.
      min_qv : float, optional
          The minimum quality value threshold for filtering transcripts.

      Returns:
      --------
      :
      dd.DataFrame
          The filtered Dask DataFrame.



