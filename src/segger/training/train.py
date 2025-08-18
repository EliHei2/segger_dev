import os
import torch
import torchmetrics
from typing import Any
from torchmetrics import F1Score
from lightning import LightningModule
from segger.models.segger_model import *
import torch.nn.functional as F

class LitSegger(LightningModule):
    """
    LitSegger is a PyTorch Lightning module for training and validating the
    Segger model.
    """

    def __init__(
        self,
        model: Segger,
        learning_rate: float = 1e-3,
        align_loss: bool = False,
        align_lambda: float = 0,
        cycle_length: int = 1000,    # Steps per cycle (e.g., 1000 steps)
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
        self.align_loss = align_loss
        self.align_lambda = align_lambda
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion_align = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.0))
        self.cycle_length = cycle_length
        self.validation_step_outputs = []


    def get_cosine_weight(self, step: int) -> float:
        """Compute cyclic weight for align_loss using cosine scheduling."""
        # Cosine varies between 0 and align_lambda
        weight = (1 + torch.cos(torch.tensor(2 * torch.pi * step / self.cycle_length))) / 2
        return weight


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
        z = self.model(batch.x_dict, batch.edge_index_dict)
        edge_index = batch["tx", "belongs", "bd"].edge_label_index
        # Compute edge scores via message passing
        out_values = (z["tx"][edge_index[0]] * z["bd"][edge_index[1]]).sum(-1)
        edge_label = batch["tx", "belongs", "bd"].edge_label
        loss = self.criterion(out_values, edge_label)
        # Log the training loss
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        if self.align_loss:   
            edge_index = batch["tx", "attracts", "tx"].edge_label_index
            edge_label = batch["tx", "attracts", "tx"].edge_label.float()
            pos_weight = (edge_label == 0).sum() / (edge_label == 1).sum()
            self.log("pos_weight", pos_weight, prog_bar=True, batch_size=batch.num_graphs)
            z_tx = z["tx"]
            out_values = (z_tx[edge_index[0]] * z_tx[edge_index[1]]).sum(-1)
            align_loss = self.criterion_align(out_values, edge_label)
            self.log("align_max", torch.max(out_values), prog_bar=True, batch_size=batch.num_graphs)
            self.log("align_min", torch.min(out_values), prog_bar=True, batch_size=batch.num_graphs)
            self.log("align_loss", align_loss, prog_bar=True, batch_size=batch.num_graphs)
            current_step = self.global_step
            align_weight = self.get_cosine_weight(current_step)
            # self.log("align_weight", align_weight, prog_bar=True, batch_size=batch.num_graphs)
            loss =  self.align_lambda * align_loss + (1-self.align_lambda) * loss
            # loss = self.align_lambda * align_loss +  loss
            # loss = align_loss
            #TOOD: cosine scheduling -- add self-loops
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
        edge_index = batch["tx", "belongs", "bd"].edge_label_index
        # Compute edge scores via message passing
        out_values = (z["tx"][edge_index[0]] * z["bd"][edge_index[1]]).sum(-1)
        
        edge_label = batch["tx", "belongs", "bd"].edge_label
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

    # def configure_optimizers(self) -> torch.optim.Optimizer:
    #     """
    #     Configures the optimizer for training.

    #     Returns
    #     -------
    #     torch.optim.Optimizer
    #         The optimizer for training.
    #     """
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)