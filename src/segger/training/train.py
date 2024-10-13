import os
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import F1Score
import lightning as L
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Metadata
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from segger.models.segger_model import *
from segger.data.utils import SpatialTranscriptomicsDataset
from typing import Any, List, Tuple, Union
from pytorch_lightning import LightningModule
import inspect


class LitSegger(LightningModule):
    """
    LitSegger is a PyTorch Lightning module for training and validating the Segger model.

    Attributes
    ----------
    model : Segger
        The Segger model wrapped with PyTorch Geometric's to_hetero for heterogeneous graph support.
    validation_step_outputs : list
        A list to store outputs from the validation steps.
    criterion : torch.nn.Module
        The loss function used for training, specifically BCEWithLogitsLoss.
    """

    def __init__(self, **kwargs):
        """
        Initializes the LitSegger module with the given parameters.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for initializing the module. Specific parameters
            depend on whether the module is initialized with new parameters or components.
        """
        super().__init__()
        new_args = inspect.getfullargspec(self.from_new)[0][1:]
        cmp_args = inspect.getfullargspec(self.from_components)[0][1:]

        # Initialize with new parameters (ensure num_tx_tokens is passed here)
        if set(kwargs.keys()) == set(new_args):
            self.from_new(**kwargs)

        # Initialize with existing components
        elif set(kwargs.keys()) == set(cmp_args):
            self.from_components(**kwargs)

        # Handle invalid arguments
        else:
            raise ValueError(
                f"Supplied kwargs do not match either constructor. Should be one of '{new_args}' or '{cmp_args}'."
            )

        self.validation_step_outputs = []
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def from_new(
        self,
        num_tx_tokens: int,
        init_emb: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        num_mid_layers: int,
        aggr: str,
        metadata: Union[Tuple, Metadata],
    ):
        """
        Initializes the LitSegger module with new parameters.

        Parameters
        ----------
        num_tx_tokens : int
            Number of unique 'tx' tokens for embedding (this must be passed here).
        init_emb : int
            Initial embedding size.
        hidden_channels : int
            Number of hidden channels.
        out_channels : int
            Number of output channels.
        heads : int
            Number of attention heads.
        aggr : str
            Aggregation method for heterogeneous graph conversion.
        num_mid_layers: int
            Number of hidden layers (excluding first and last layers).
        metadata : Union[Tuple, Metadata]
            Metadata for heterogeneous graph structure.
        """
        # Create the Segger model (ensure num_tx_tokens is passed here)
        model = Segger(
            num_tx_tokens=num_tx_tokens,  # This is required and must be passed here
            init_emb=init_emb,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            num_mid_layers=num_mid_layers,
        )
        # Convert model to handle heterogeneous graphs
        model = to_hetero(model, metadata=metadata, aggr=aggr)
        self.model = model
        # Save hyperparameters
        self.save_hyperparameters()

    def from_components(self, model: Segger):
        """
        Initializes the LitSegger module with existing Segger components.

        Parameters
        ----------
        model : Segger
            The Segger model to be used.
        """
        self.model = model

    def forward(self, batch: SpatialTranscriptomicsDataset) -> torch.Tensor:
        """
        Forward pass for the batch of data.

        Parameters
        ----------
        batch : SpatialTranscriptomicsDataset
            The batch of data, including node features and edge indices.

        Returns
        -------
        torch.Tensor
            The output of the model.
        """
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z["tx"], z["bd"].t())  # Example for bipartite graph
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
            The loss value for the current training step.
        """
        # Forward pass to get the logits
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z["tx"], z["bd"].t())

        # Get edge labels and logits
        edge_label_index = batch["tx", "belongs", "bd"].edge_label_index
        out_values = output[edge_label_index[0], edge_label_index[1]]
        edge_label = batch["tx", "belongs", "bd"].edge_label

        # Compute binary cross-entropy loss with logits (no sigmoid here)
        loss = self.criterion(out_values, edge_label)

        # Log the training loss
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines the validation step.

        Parameters
        ----------
        batch : Any
            The batch of data.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        torch.Tensor
            The loss value for the current validation step.
        """
        # Forward pass to get the logits
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z["tx"], z["bd"].t())

        # Get edge labels and logits
        edge_label_index = batch["tx", "belongs", "bd"].edge_label_index
        out_values = output[edge_label_index[0], edge_label_index[1]]
        edge_label = batch["tx", "belongs", "bd"].edge_label

        # Compute binary cross-entropy loss with logits (no sigmoid here)
        loss = self.criterion(out_values, edge_label)

        # Apply sigmoid to logits for AUROC and F1 metrics
        out_values_prob = torch.sigmoid(out_values)

        # Compute metrics
        auroc = torchmetrics.AUROC(task="binary")
        auroc_res = auroc(out_values_prob, edge_label)

        f1 = F1Score(task="binary").to(self.device)
        f1_res = f1(out_values_prob, edge_label)

        # Log validation metrics
        self.log("validation_loss", loss, batch_size=batch.num_graphs)
        self.log("validation_auroc", auroc_res, prog_bar=True, batch_size=batch.num_graphs)
        self.log("validation_f1", f1_res, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
