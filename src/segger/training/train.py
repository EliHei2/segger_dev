import os
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import F1Score
import lightning as L
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

    def __init__(self, learning_rate: float = 1e-3, **kwargs):
        """
        Initializes the LitSegger module with the given parameters.

        Parameters
        ----------
        learning_rate : float
            The learning rate for the optimizer.
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
        self.learning_rate = learning_rate

    def from_new(
        self,
        num_node_features: dict[str, int],
        init_emb: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        num_mid_layers: int,
        aggr: str,
        is_token_based: bool = True,
    ):
        """
        Initializes the LitSegger module with new parameters.

        Parameters
        ----------
        num_node_features : dict[str, int]
            Number of node features for each node type.
        init_emb : int
            Initial embedding size.
        hidden_channels : int
            Number of hidden channels.
        out_channels : int
            Number of output channels.
        heads : int
            Number of attention heads.
        num_mid_layers: int
            Number of hidden layers (excluding first and last layers).
        aggr : str
            Aggregation method for heterogeneous graph conversion.
        is_token_based : bool
            Whether the model is using token-based embeddings or scRNAseq embeddings.
        """
        # Create the Segger model (ensure num_tx_tokens is passed here)
        self.model = Segger(
            num_node_features=num_node_features,
            init_emb=init_emb,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            num_mid_layers=num_mid_layers,
            is_token_based=is_token_based,
        )
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
        edge_label_index = batch["tx", "belongs", "bd"].edge_label_index
        output = self.model.decode(z, edge_label_index)
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
        # Get edge labels
        edge_label_index = batch["tx", "belongs", "bd"].edge_label_index
        edge_label = batch["tx", "belongs", "bd"].edge_label

        # Forward pass to get the logits
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = self.model.decode(z, edge_label_index)

        # Compute binary cross-entropy loss with logits (no sigmoid here)
        loss = self.criterion(output, edge_label)

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
        # Get edge labels
        edge_label_index = batch["tx", "belongs", "bd"].edge_label_index
        edge_label = batch["tx", "belongs", "bd"].edge_label

        # Forward pass to get the logits
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = self.model.decode(z, edge_label_index)

        # Compute binary cross-entropy loss with logits (no sigmoid here)
        loss = self.criterion(output, edge_label)

        # Apply sigmoid to logits for AUROC and F1 metrics
        out_values_prob = torch.sigmoid(output)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
