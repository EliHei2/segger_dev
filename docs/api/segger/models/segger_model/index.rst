segger.models.segger_model
==========================

.. py:module:: segger.models.segger_model




Module Contents
---------------

.. py:class:: Segger(init_emb: int = 16, hidden_channels: int = 32, num_mid_layers: int = 3, out_channels: int = 32, heads: int = 3)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


   .. py:attribute:: num_mid_layers


   .. py:attribute:: lin0


   .. py:attribute:: conv_first


   .. py:attribute:: lin_first


   .. py:attribute:: conv_last


   .. py:attribute:: lin_last


   .. py:method:: forward(x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor

      Forward pass for the Segger model.

      :param x: Node features.
      :type x: Tensor
      :param edge_index: Edge indices.
      :type edge_index: EdgeIndex

      :returns: Output node embeddings.
      :rtype: Tensor



   .. py:method:: decode(z: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor

      Decode the node embeddings to predict edge values.

      :param z: Node embeddings.
      :type z: Tensor
      :param edge_label_index: Edge label indices.
      :type edge_label_index: EdgeIndex

      :returns: Predicted edge values.
      :rtype: Tensor



