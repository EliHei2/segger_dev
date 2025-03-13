segger.data.utils
=================

.. py:module:: segger.data.utils






Module Contents
---------------

.. py:function:: try_import(module_name)

.. py:function:: uint32_to_str(cell_id_uint32: int, dataset_suffix: str) -> str

   Convert a 32-bit unsigned integer cell ID to a string with a specific suffix.

   Parameters:
   cell_id_uint32 (int): The 32-bit unsigned integer cell ID.
   dataset_suffix (str): The suffix to append to the string representation of the cell ID.

   Returns:
   str: The string representation of the cell ID with the appended suffix.


.. py:function:: filter_transcripts(transcripts_df: pandas.DataFrame, min_qv: float = 20.0) -> pandas.DataFrame

   Filters transcripts based on quality value and removes unwanted transcripts.

   Parameters:
   transcripts_df (pd.DataFrame): The dataframe containing transcript data.
   min_qv (float): The minimum quality value threshold for filtering transcripts.

   Returns:
   pd.DataFrame: The filtered dataframe.


.. py:function:: compute_transcript_metrics(df: pandas.DataFrame, qv_threshold: float = 30, cell_id_col: str = 'cell_id') -> Dict[str, Any]

   Computes various metrics for a given dataframe of transcript data filtered by quality value threshold.

   Parameters:
   df (pd.DataFrame): The dataframe containing transcript data.
   qv_threshold (float): The quality value threshold for filtering transcripts.
   cell_id_col (str): The name of the column representing the cell ID.

   Returns:
   Dict[str, Any]: A dictionary containing:
       - 'percent_assigned' (float): The percentage of assigned transcripts.
       - 'percent_cytoplasmic' (float): The percentage of cytoplasmic transcripts among assigned transcripts.
       - 'percent_nucleus' (float): The percentage of nucleus transcripts among assigned transcripts.
       - 'percent_non_assigned_cytoplasmic' (float): The percentage of non-assigned cytoplasmic transcripts among all non-assigned transcripts.
       - 'gene_metrics' (pd.DataFrame): A dataframe containing gene-level metrics:
           - 'feature_name': The gene name.
           - 'percent_assigned': The percentage of assigned transcripts for each gene.
           - 'percent_cytoplasmic': The percentage of cytoplasmic transcripts for each gene.


.. py:function:: create_anndata(df: pandas.DataFrame, panel_df: Optional[pandas.DataFrame] = None, min_transcripts: int = 5, cell_id_col: str = 'cell_id', qv_threshold: float = 30, min_cell_area: float = 10.0, max_cell_area: float = 1000.0) -> anndata.AnnData

   Generates an AnnData object from a dataframe of segmented transcriptomics data.

   Parameters:
   df (pd.DataFrame): The dataframe containing segmented transcriptomics data.
   panel_df (Optional[pd.DataFrame]): The dataframe containing panel information.
   min_transcripts (int): The minimum number of transcripts required for a cell to be included.
   cell_id_col (str): The column name representing the cell ID in the input dataframe.
   qv_threshold (float): The quality value threshold for filtering transcripts.
   min_cell_area (float): The minimum cell area to include a cell.
   max_cell_area (float): The maximum cell area to include a cell.

   Returns:
   ad.AnnData: The generated AnnData object containing the transcriptomics data and metadata.


.. py:class:: BuildTxGraph(r: float, loop: bool = False, max_num_neighbors: int = 32, flow: str = 'source_to_target', num_workers: int = 5)

   Bases: :py:obj:`torch_geometric.transforms.BaseTransform`


   .. py:attribute:: r


   .. py:attribute:: loop


   .. py:attribute:: max_num_neighbors


   .. py:attribute:: flow


   .. py:attribute:: num_workers


   .. py:method:: forward(data: torch_geometric.data.HeteroData) -> torch_geometric.data.HeteroData


.. py:function:: calculate_gene_celltype_abundance_embedding(adata: anndata.AnnData, celltype_column: str) -> pandas.DataFrame

   Calculate the cell type abundance embedding for each gene based on the percentage of cells in each cell type
   that express the gene (non-zero expression).

   Parameters:
   -----------
   adata : AnnData
       An AnnData object containing gene expression data and cell type information.
   celltype_column : str
       The column name in `adata.obs` that contains the cell type information.

   Returns:
   --------
   :
   pd.DataFrame
       A DataFrame where rows are genes and columns are cell types, with each value representing
       the percentage of cells in that cell type expressing the gene.

   Example:
   --------
   >>> adata = AnnData(...)  # Load your scRNA-seq AnnData object
   >>> celltype_column = 'celltype_major'
   >>> abundance_df = calculate_gene_celltype_abundance_embedding(adata, celltype_column)
   >>> abundance_df.head()


.. py:function:: get_edge_index(coords_1: numpy.ndarray, coords_2: numpy.ndarray, k: int = 5, dist: int = 10, method: str = 'kd_tree', gpu: bool = False, workers: int = 1) -> torch.Tensor

   Computes edge indices using various methods (KD-Tree, FAISS, RAPIDS cuML, cuGraph, or cuSpatial).

   Parameters:
   -----------
   coords_1 : np.ndarray
       First set of coordinates.
   coords_2 : np.ndarray
       Second set of coordinates.
   k : int, optional
       Number of nearest neighbors.
   dist : int, optional
       Distance threshold.
   method : str, optional
       The method to use ('kd_tree', 'faiss', 'rapids', 'cugraph', 'cuspatial').
   gpu : bool, optional
       Whether to use GPU acceleration (applicable for FAISS).

   Returns:
   --------
   :
   torch.Tensor
       Edge indices.


.. py:function:: get_edge_index_kdtree(coords_1: numpy.ndarray, coords_2: numpy.ndarray, k: int = 5, dist: int = 10, workers: int = 1) -> torch.Tensor

   Computes edge indices using KDTree.

   Parameters:
   -----------
   coords_1 : np.ndarray
       First set of coordinates.
   coords_2 : np.ndarray
       Second set of coordinates.
   k : int, optional
       Number of nearest neighbors.
   dist : int, optional
       Distance threshold.

   Returns:
   --------
   :
   torch.Tensor
       Edge indices.


.. py:function:: get_edge_index_faiss(coords_1: numpy.ndarray, coords_2: numpy.ndarray, k: int = 5, dist: int = 10, gpu: bool = False) -> torch.Tensor

   Computes edge indices using FAISS.

   Parameters:
   -----------
   coords_1 : np.ndarray
       First set of coordinates.
   coords_2 : np.ndarray
       Second set of coordinates.
   k : int, optional
       Number of nearest neighbors.
   dist : int, optional
       Distance threshold.
   gpu : bool, optional
       Whether to use GPU acceleration.

   Returns:
   --------
   :
   torch.Tensor
       Edge indices.


.. py:function:: get_edge_index_rapids(coords_1: numpy.ndarray, coords_2: numpy.ndarray, k: int = 5, dist: int = 10) -> torch.Tensor

   Computes edge indices using RAPIDS cuML.

   Parameters:
   -----------
   coords_1 : np.ndarray
       First set of coordinates.
   coords_2 : np.ndarray
       Second set of coordinates.
   k : int, optional
       Number of nearest neighbors.
   dist : int, optional
       Distance threshold.

   Returns:
   --------
   :
   torch.Tensor
       Edge indices.


.. py:function:: get_edge_index_cugraph(coords_1: numpy.ndarray, coords_2: numpy.ndarray, k: int = 5, dist: int = 10) -> torch.Tensor

   Computes edge indices using RAPIDS cuGraph.

   Parameters:
   -----------
   coords_1 : np.ndarray
       First set of coordinates.
   coords_2 : np.ndarray
       Second set of coordinates.
   k : int, optional
       Number of nearest neighbors.
   dist : int, optional
       Distance threshold.

   Returns:
   --------
   :
   torch.Tensor
       Edge indices.


.. py:function:: get_edge_index_cuspatial(coords_1: numpy.ndarray, coords_2: numpy.ndarray, k: int = 5, dist: int = 10) -> torch.Tensor

   Computes edge indices using cuSpatial's spatial join functionality.

   Parameters:
   -----------
   coords_1 : np.ndarray
       First set of coordinates (2D).
   coords_2 : np.ndarray
       Second set of coordinates (2D).
   k : int, optional
       Number of nearest neighbors.
   dist : int, optional
       Distance threshold.

   Returns:
   --------
   :
   torch.Tensor
       Edge indices.


.. py:class:: SpatialTranscriptomicsDataset(root: str, transform: Callable = None, pre_transform: Callable = None, pre_filter: Callable = None)

   Bases: :py:obj:`torch_geometric.data.InMemoryDataset`


   A dataset class for handling SpatialTranscriptomics spatial transcriptomics data.

   .. attribute:: root

      The root directory where the dataset is stored.

      :type: str

   .. attribute:: transform

      A function/transform that takes in a Data object and returns a transformed version.

      :type: callable

   .. attribute:: pre_transform

      A function/transform that takes in a Data object and returns a transformed version.

      :type: callable

   .. attribute:: pre_filter

      A function that takes in a Data object and returns a boolean indicating whether to keep it.

      :type: callable


   .. py:property:: raw_file_names
      :type: List[str]

      Return a list of raw file names in the raw directory.

      :returns: List of raw file names.
      :rtype: List[str]


   .. py:property:: processed_file_names
      :type: List[str]

      Return a list of processed file names in the processed directory.

      :returns: List of processed file names.
      :rtype: List[str]


   .. py:method:: download() -> None

      Download the raw data. This method should be overridden if you need to download the data.



   .. py:method:: process() -> None

      Process the raw data and save it to the processed directory. This method should be overridden if you need to process the data.



   .. py:method:: len() -> int

      Return the number of processed files.

      :returns: Number of processed files.
      :rtype: int



   .. py:method:: get(idx: int) -> torch_geometric.data.Data

      Get a processed data object.

      :param idx: Index of the data object to retrieve.
      :type idx: int

      :returns: The processed data object.
      :rtype: Data



.. py:function:: get_edge_index_hnsw(coords_1: numpy.ndarray, coords_2: numpy.ndarray, k: int = 5, dist: int = 10) -> torch.Tensor

   Computes edge indices using the HNSW algorithm.

   Parameters:
   -----------
   coords_1 : np.ndarray
       First set of coordinates.
   coords_2 : np.ndarray
       Second set of coordinates.
   k : int, optional
       Number of nearest neighbors.
   dist : int, optional
       Distance threshold.

   Returns:
   --------
   :
   torch.Tensor
       Edge indices.


