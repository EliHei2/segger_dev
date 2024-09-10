segger.training.train
=====================

.. py:module:: segger.training.train




Module Contents
---------------

.. py:class:: LitSegger(**kwargs)

   Bases: :py:obj:`pytorch_lightning.LightningModule`


   LitSegger is a PyTorch Lightning module for training and validating the Segger model.

   This class handles the training loop, validation loop, and optimizer configuration
   for the Segger model, which is designed for heterogeneous graph data. The model is
   compatible with large-scale datasets and leverages PyTorch Geometric for graph-related operations.

   .. attribute:: model

      The Segger model wrapped with PyTorch Geometric's `to_hetero` for heterogeneous graph support.

      :type: Segger

   .. attribute:: validation_step_outputs

      A list to store outputs from the validation steps.

      :type: list

   .. attribute:: criterion

      The loss function used for training, specifically `BCEWithLogitsLoss`.

      :type: torch.nn.Module

   .. method:: forward(batch: SpatialTranscriptomicsSample) -> torch.Tensor

      Forward pass for the batch of data.

   .. method:: training_step(batch: Any, batch_idx: int) -> torch.Tensor

      Defines a single training step.

   .. method:: validation_step(batch: Any, batch_idx: int) -> torch.Tensor

      Defines a single validation step.

   .. method:: configure_optimizers() -> torch.optim.Optimizer

      Configures the optimizer and learning rate scheduler.



   .. py:attribute:: new_args


   .. py:attribute:: cmp_args


   .. py:attribute:: validation_step_outputs
      :value: []



   .. py:attribute:: criterion


   .. py:method:: from_new(init_emb: int, hidden_channels: int, out_channels: int, heads: int, aggr: str, metadata: Union[Tuple, torch_geometric.typing.Metadata])

      Initializes the LitSegger module with new parameters.

      :param init_emb: Initial embedding size.
      :type init_emb: int
      :param hidden_channels: Number of hidden channels.
      :type hidden_channels: int
      :param out_channels: Number of output channels.
      :type out_channels: int
      :param heads: Number of attention heads in GAT layers.
      :type heads: int
      :param aggr: Aggregation method for the `to_hetero` conversion.
      :type aggr: str
      :param metadata: Metadata for the heterogeneous graph, used to define the node and edge types.
      :type metadata: Union[Tuple, Metadata]

      .. rubric:: Notes

      This method creates a new Segger model with the specified parameters and
      wraps it in a `to_hetero` call to handle heterogeneous graph structures.



   .. py:method:: from_components(model: segger.models.segger_model.Segger)

      Initializes the LitSegger module with existing components for testing.

      :param model: The Segger model to be used.
      :type model: Segger

      .. rubric:: Notes

      This method directly assigns the provided Segger model to the LitSegger
      module, which can be useful for testing or reusing pre-trained models.



   .. py:method:: forward(batch: segger.data.utils.SpatialTranscriptomicsSample) -> torch.Tensor

      Forward pass for the batch of data.

      This method computes the forward pass of the model for a batch of data, producing
      output tensors based on the input features and edge indices.

      :param batch: Batch of data, including node features and edge indices.
      :type batch: SpatialTranscriptomicsSample

      :returns: The output tensor resulting from the forward pass.
      :rtype: torch.Tensor



   .. py:method:: training_step(batch: Any, batch_idx: int) -> torch.Tensor

      Defines the training step.

      This method computes the loss for a single batch during training, logs the
      loss value, and returns it for backpropagation.

      :param batch: The batch of data.
      :type batch: Any
      :param batch_idx: The index of the batch.
      :type batch_idx: int

      :returns: The loss value for the current training step.
      :rtype: torch.Tensor



   .. py:method:: validation_step(batch: Any, batch_idx: int) -> torch.Tensor

      Defines the validation step.

      This method computes the loss for a single batch during validation, logs the
      loss and other metrics (AUROC, F1 score), and returns the loss.

      :param batch: The batch of data.
      :type batch: Any
      :param batch_idx: The index of the batch.
      :type batch_idx: int

      :returns: The loss value for the current validation step.
      :rtype: torch.Tensor



   .. py:method:: configure_optimizers() -> torch.optim.Optimizer

      Configures the optimizer.

      This method defines and returns the optimizer to be used during training,
      along with any learning rate schedulers if needed.

      :returns: The optimizer used for training the model.
      :rtype: torch.optim.Optimizer



