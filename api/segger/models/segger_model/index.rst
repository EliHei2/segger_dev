segger.models.segger_model
==========================

.. py:module:: segger.models.segger_model




Module Contents
---------------

.. py:class:: Segger(num_tx_tokens: int, init_emb: int = 16, hidden_channels: int = 32, num_mid_layers: int = 3, out_channels: int = 32, heads: int = 3)

   Bases: :py:obj:`torch.nn.Module`


   .. py:attribute:: tx_embedding


   .. py:attribute:: lin0


   .. py:attribute:: conv_first


   .. py:attribute:: num_mid_layers


   .. py:attribute:: conv_last


   .. py:method:: forward(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor

      Forward pass for the Segger model.

      :param x: Node features.
      :type x: Tensor
      :param edge_index: Edge indices.
      :type edge_index: Tensor

      :returns: Output node embeddings.
      :rtype: Tensor



   .. py:method:: decode(z: torch.Tensor, edge_index: Union[torch.Tensor, torch_sparse.SparseTensor]) -> torch.Tensor

      Decode the node embeddings to predict edge values.

      :param z: Node embeddings.
      :type z: Tensor
      :param edge_index: Edge label indices.
      :type edge_index: EdgeIndex

      :returns: Predicted edge values.
      :rtype: Tensor



