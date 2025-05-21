from typing import Any, Tuple, Optional
from torchmetrics import F1Score, AUROC
from lightning import LightningModule
import pandas as pd
import torch
import os

from ..data.universal._enum import FEATURE_INDEX_FILE
from ..models.segger_model import *
from ..config.config import (
    _check_gene_embedding_indices,
    _check_gene_embedding_weights,
    SeggerConfig
)

class LitSegger(LightningModule):
    """
    LitSegger is a PyTorch Lightning module for training and validating the 
    Segger model.
    """

    def __init__(
        self,
        gene_embedding_indices: os.PathLike,
        gene_embedding_weights: Optional[os.PathLike] = None,
        in_channels: int = 16,
        hidden_channels: int = 32,
        out_channels: int = 32,
        n_mid_layers: int = 3,
        n_heads: int = 3,
        learning_rate: float = 1e-3,
    ):
        #TODO: Add documentation
        super().__init__()
        self.save_hyperparameters()
        n_genes, embedding_weights = LitSegger._get_gene_embedding(
            gene_embedding_indices,
            gene_embedding_weights,
        )
        self.model = Segger(
            n_genes=n_genes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_mid_layers=n_mid_layers,
            n_heads=n_heads,
            embedding_weights=embedding_weights,
        )
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
        edge_label_index = batch["tx", "belongs", "bd"].edge_label_index
        output = self.model.decode(z, edge_label_index)
        return output

    def get_bce_loss(self, batch: Any) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss for the given batch.

        Parameters
        ----------
        batch : Any
            The input batch containing node features, edge indices,
            and edge labels.

        Returns
        -------
        torch.Tensor
            Computed binary cross-entropy loss.
        """
        # Forward pass to get the logits
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z['tx'], z['bd'].t())

        # Get edge labels and logits
        pos_index = batch['tx', 'belongs', 'bd'].edge_index
        neg_index = batch['tx', 'belongs', 'bd'].neg_edge_index
        pos_out_values = output[pos_index[0], pos_index[1]]
        neg_out_values = output[neg_index[0], neg_index[1]]
        out_values = torch.concat((pos_out_values, neg_out_values))
        edge_label = torch.concat((
            torch.ones_like(pos_out_values),
            torch.zeros_like(neg_out_values)
        ))
        
        # Compute binary cross-entropy loss with logits (no sigmoid here)
        return self.criterion(out_values, edge_label)


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Parameters
        ----------
        batch : Any
            The input batch containing node features, edge indices, 
            and edge labels.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Computed training loss for the current batch.
        """
        # Get loss
        loss = self.get_bce_loss(batch)

        # Log the training loss
        self.log(
            "training_loss",
            loss,
            prog_bar=True,
            batch_size=batch.num_graphs
        )

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
        # Get loss
        loss = self.get_bce_loss(batch)

        # Apply sigmoid to logits for AUROC and F1 metrics
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = torch.matmul(z['tx'], z['bd'].t())
        pos_index = batch['tx', 'belongs', 'bd'].edge_index
        neg_index = batch['tx', 'belongs', 'bd'].neg_edge_index
        pos_out_values = output[pos_index[0], pos_index[1]]
        neg_out_values = output[neg_index[0], neg_index[1]]
        out_values = torch.concat((pos_out_values, neg_out_values))
        edge_label = torch.concat((
            torch.ones_like(pos_out_values),
            torch.zeros_like(neg_out_values)
        ))
        probs = torch.sigmoid(out_values)

        # Compute metrics
        auroc = AUROC(task="binary").to(self.device)(probs, edge_label)
        f1_score = F1Score(task="binary").to(self.device)(probs, edge_label)

        # Log validation metrics
        shared_kwargs = dict(prog_bar=True, batch_size=batch.num_graphs)
        self.log("validation_loss", loss, **shared_kwargs)
        self.log("validation_auroc", auroc, **shared_kwargs)
        self.log("validation_f1", f1_score, **shared_kwargs)

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
    
    @staticmethod
    def _get_gene_embedding(
        indices_path: os.PathLike,
        weights_path: Optional[os.PathLike] = None,
    ) -> Tuple[int, Optional[Tensor]]:
        """
        Loads gene embedding indices and weights.

        Parameters
        ----------
        index_path : os.PathLike
            Path to CSV file containing the ordered list of gene indices.
        weight_path : os.PathLike, optional
            Path to CSV file containing gene embedding weights, indexed by gene.

        Returns
        -------
        int
            Number of genes.
        Tensor or None
            Tensor of embedding weights if provided, else None.
        """
        _check_gene_embedding_indices(indices_path)
        indices = pd.read_csv(indices_path)

        if weights_path is not None:
            _check_gene_embedding_weights(weights_path)
            weights = pd.read_csv(weights_path, index_col=0)
            reordered = weights.loc[indices.values.flatten()]
            return indices.shape[0], torch.from_numpy(reordered.values)
        else:
            return indices.shape[0], None
