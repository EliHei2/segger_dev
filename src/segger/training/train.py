import os
import torch
import torchmetrics
from typing import Any
from torchmetrics import F1Score
from lightning import LightningModule
from segger.models.segger_model import *


class LitSegger(LightningModule):
    """
    LitSegger is a PyTorch Lightning module for training and validating the
    Segger model.
    """

    def __init__(
        self,
        model: Segger,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize the Segger training module.

        Parameters
        ----------
        model : Segger
            The Segger model to be trained.
        learning_rate : float, optional
            Learning rate for the optimizer. Default is 1e-3.
        """
        super().__init__()
        # Set model and store initialization parameters for reproducibility
        self.model = model
        self.save_hyperparameters()

        # Other setup
        self.learning_rate = learning_rate
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []

    def forward(self, batch) -> torch.Tensor:
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
        z, _ = self.model(batch.x_dict, batch.edge_index_dict)
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
        z, _ = self.model(batch.x_dict, batch.edge_index_dict)
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
        self.log(
            "validation_auroc", auroc_res, prog_bar=True, batch_size=batch.num_graphs
        )
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
