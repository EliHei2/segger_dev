segger.prediction.predict
=========================

.. py:module:: segger.prediction.predict




Module Contents
---------------

.. py:function:: load_model(checkpoint_path: str, init_emb: int, hidden_channels: int, out_channels: int, heads: int, aggr: str) -> segger.training.train.LitSegger

   Load a LitSegger model from a checkpoint.

   :param checkpoint_path: Specific checkpoint file to load, or directory where the model
                           checkpoints are stored. If directory, the latest checkpoint is loaded.
   :type checkpoint_path: os.Pathlike

   :returns: The loaded LitSegger model.
   :rtype: LitSegger

   :raises FileNotFoundError: If the specified checkpoint file does not exist.


.. py:function:: get_similarity_scores(model: segger.models.segger_model.Segger, batch: torch_geometric.data.Batch, from_type: str, to_type: str)

   Compute similarity scores between 'from_type' and 'to_type' embeddings
   within a batch.

   :param model: The segmentation model used to generate embeddings.
   :type model: Segger
   :param batch: A batch of data containing input features and edge indices.
   :type batch: Batch
   :param from_type: The type of node from which the similarity is computed.
   :type from_type: str
   :param to_type: The type of node to which the similarity is computed.
   :type to_type: str

   :returns: A dense tensor containing the similarity scores between 'from_type'
             and 'to_type' nodes.
   :rtype: torch.Tensor


.. py:function:: predict_batch(lit_segger: segger.training.train.LitSegger, batch: torch_geometric.data.Batch, score_cut: float, use_cc: bool = True) -> pandas.DataFrame

   Predict cell assignments for a batch of transcript data using a
   segmentation model.

   :param lit_segger: The lightning module wrapping the segmentation model.
   :type lit_segger: LitSegger
   :param batch: A batch of transcript and cell data.
   :type batch: Batch
   :param score_cut: The threshold for assigning transcripts to cells based on similarity
                     scores.
   :type score_cut: float

   :returns: A DataFrame containing the transcript IDs, similarity scores, and
             assigned cell IDs.
   :rtype: pd.DataFrame


.. py:function:: predict(lit_segger: segger.training.train.LitSegger, data_loader: torch_geometric.loader.DataLoader, score_cut: float, use_cc: bool = True) -> pandas.DataFrame

   Predict cell assignments for multiple batches of transcript data using
   a segmentation model.

   :param lit_segger: The lightning module wrapping the segmentation model.
   :type lit_segger: LitSegger
   :param data_loader: A data loader providing batches of transcript and cell data.
   :type data_loader: DataLoader
   :param score_cut: The threshold for assigning transcripts to cells based on similarity
                     scores.
   :type score_cut: float

   :returns: A DataFrame containing the transcript IDs, similarity scores, and
             assigned cell IDs, consolidated across all batches.
   :rtype: pd.DataFrame


