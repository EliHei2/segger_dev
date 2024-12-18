segger.training.train
=====================

.. py:module:: segger.training.train




Module Contents
---------------

.. py:class:: LitSegger(**kwargs)

   Bases: :py:obj:`lightning.LightningModule`


   LitSegger is a PyTorch Lightning module for training and validating the Segger model.

   .. attribute:: model

      The Segger model wrapped with PyTorch Geometric's to_hetero for heterogeneous graph support.

      :type: Segger

   .. attribute:: validation_step_outputs

      A list to store outputs from the validation steps.

      :type: list

   .. attribute:: criterion

      The loss function used for training, specifically BCEWithLogitsLoss.

      :type: torch.nn.Module


   .. py:attribute:: new_args


   .. py:attribute:: cmp_args


   .. py:attribute:: validation_step_outputs
      :value: []



   .. py:attribute:: criterion


   .. py:method:: from_new(num_tx_tokens: int, init_emb: int, hidden_channels: int, out_channels: int, heads: int, aggr: str, metadata: Union[Tuple, torch_geometric.typing.Metadata])

      Initializes the LitSegger module with new parameters.

      :param num_tx_tokens: Number of unique 'tx' tokens for embedding (this must be passed here).
      :type num_tx_tokens: int
      :param init_emb: Initial embedding size.
      :type init_emb: int
      :param hidden_channels: Number of hidden channels.
      :type hidden_channels: int
      :param out_channels: Number of output channels.
      :type out_channels: int
      :param heads: Number of attention heads.
      :type heads: int
      :param aggr: Aggregation method for heterogeneous graph conversion.
      :type aggr: str
      :param metadata: Metadata for heterogeneous graph structure.
      :type metadata: Union[Tuple, Metadata]



   .. py:method:: from_components(model: Segger)

      Initializes the LitSegger module with existing Segger components.

      :param model: The Segger model to be used.
      :type model: Segger



   .. py:method:: forward(batch: segger.data.utils.SpatialTranscriptomicsDataset) -> torch.Tensor

      Forward pass for the batch of data.

      :param batch: The batch of data, including node features and edge indices.
      :type batch: SpatialTranscriptomicsDataset

      :returns: The output of the model.
      :rtype: torch.Tensor



   .. py:method:: training_step(batch: Any, batch_idx: int) -> torch.Tensor

      Defines the training step.

      :param batch: The batch of data.
      :type batch: Any
      :param batch_idx: The index of the batch.
      :type batch_idx: int

      :returns: The loss value for the current training step.
      :rtype: torch.Tensor



   .. py:method:: validation_step(batch: Any, batch_idx: int) -> torch.Tensor

      Defines the validation step.

      :param batch: The batch of data.
      :type batch: Any
      :param batch_idx: The index of the batch.
      :type batch_idx: int

      :returns: The loss value for the current validation step.
      :rtype: torch.Tensor



   .. py:method:: configure_optimizers() -> torch.optim.Optimizer

      Configures the optimizer for training.

      :returns: The optimizer for training.
      :rtype: torch.optim.Optimizer



