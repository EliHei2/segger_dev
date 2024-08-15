import os
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import F1Score
from lightning import LightningModule
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Metadata
from torch_geometric.nn import to_hetero
from segger.models.segger_model import Segger
from segger.data.utils import XeniumDataset
from typing import Any, List, Tuple, Union


class LitSegger(LightningModule):
    """
    LitSegger is a PyTorch Lightning module for training and validating the 
    Segger model.

    Attributes
    ----------
    model : Segger
        The Segger model wrapped with PyTorch Geometric's to_hetero for 
        heterogeneous graph support.
    validation_step_outputs : list
        A list to store outputs from the validation steps.
    criterion : torch.nn.Module
        The loss function used for training, specifically BCEWithLogitsLoss.

    Methods
    -------
    __init__(init_emb, hidden_channels, out_channels, heads, aggr, metadata)
        Initializes the LitSegger module with the given parameters.
    forward(batch)
        Forward pass of the model.
    training_step(batch, batch_idx)
        Defines a single training step.
    validation_step(batch, batch_idx)
        Defines a single validation step.
    configure_optimizers()
        Configures the optimizers and learning rate scheduler.
    """
    def __init__(
        self, 
        init_emb: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        aggr: str,
        metadata: Union[Tuple, Metadata],
    ):
        """
        Initializes the LitSegger module for training and validation.

        Parameters
        ----------
        init_emb : int
            Initial embedding size.
        hidden_channels : int
            Number of hidden channels.
        out_channels : int
            Number of output channels.
        heads : int
            Number of attention heads.
        aggr : str
            Aggregation method.
        metadata : Union[Tuple, Metadata]
            Metadata for the heterogeneous graph.
        """
        # Create Segger model
        super().__init__()
        model = Segger(
            init_emb=init_emb,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
        )
        model = to_hetero(model, metadata=metadata, aggr=aggr)
        self.model = model
        # Save hyperparameters to file for reconstruction/record-keeping
        self.save_hyperparameters()
        # Other setup
        self.validation_step_outputs = []
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch: XeniumDataset):
        """
        Forward pass for the batch of data.

        Parameters
        ----------
        batch : XeniumDataset
            Batch of data.

        Returns
        -------
        torch.Tensor
            Output of the model.
        """
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z["tx"], z["nc"].t())
        return output

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines the training step.

        Parameters
        ----------
        batch : Any
            The batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z["tx"], z["nc"].t())
        edge_label_index = batch["tx", "belongs", "nc"].edge_label_index
        out_values = output[edge_label_index[0], edge_label_index[1]]
        edge_label = batch["tx", "belongs", "nc"].edge_label
        loss = self.criterion(out_values, edge_label)
        self.log(
            "train_loss", loss, prog_bar=False, batch_size=batch.num_graphs
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines a single validation step.

        Parameters
        ----------
        batch : Any
            A batch of data.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            A tensor containing the validation loss.
        """
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z["tx"], z["nc"].t())
        edge_label_index = batch["tx", "belongs", "nc"].edge_label_index
        out_values = output[edge_label_index[0], edge_label_index[1]]
        edge_label = batch["tx", "belongs", "nc"].edge_label
        loss = self.criterion(out_values, edge_label)
        out_values = F.sigmoid(out_values)
        auroc = torchmetrics.AUROC(task="binary")
        auroc_res = auroc(out_values, edge_label)
        f1 = F1Score(task="binary", num_classes=2).to(self.device)
        f1_res = f1(out_values, edge_label)
        self.log("validation_loss", loss, batch_size=batch.num_graphs)
        self.log(
            "validation_auroc",
            auroc_res,
            prog_bar=False,
            batch_size=batch.num_graphs,
        )
        self.log(
            "validation_f1", f1_res, prog_bar=False, batch_size=batch.num_graphs
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizers and learning rate scheduler.

        Returns
        -------
            torch.optim.Optimizer: The optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
