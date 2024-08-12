import os
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import F1Score
import lightning as L
from torch_geometric.loader import DataLoader
from segger.models.segger_model import Segger
from segger.data.utils import XeniumDataset
from typing import Any, List


class LitSegger(L.LightningModule):
    def __init__(self, model: Segger):
        """
        Initializes the LitSegger module for training and validation.

        Args:
            model (Segger): The Segger model.
        """
        super().__init__()
        self.model = model
        self.validation_step_outputs = []
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch: XeniumDataset):
        """
        Forward pass for the batch of data.

        Args:
            batch (XeniumDataset): Batch of data.

        Returns:
            Output of the model.
        """
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z["tx"], z["nc"].t())
        return output

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines the training step.

        Args:
            batch (Any): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
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
        Defines the validation step.

        Args:
            batch (Any): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
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
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
