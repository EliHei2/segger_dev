import torch
from typing import Any
from lightning import LightningModule
from torchmetrics import F1Score, AUROC
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
        self.save_hyperparameters(self.model.hparams)

        # Other setup
        self.learning_rate = learning_rate
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []

    def on_load_checkpoint(self, checkpoint):
        return super().on_load_checkpoint(checkpoint)

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
        '''
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
        loss = self.criterion(out_values, edge_label)
        
        # Log the training loss
        self.log("training_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
        return loss
        '''
        # Get edge labels
        edge_label_index = batch['tx', 'belongs', 'bd'].edge_label_index
        edge_label = batch['tx', 'belongs', 'bd'].edge_label

        # Forward pass to get the logits
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = self.model.decode(z, edge_label_index)

        # Compute binary cross-entropy loss with logits (sigmoid in function)
        loss = self.criterion(output, edge_label)
        return loss


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
        edge_label = batch['tx', 'belongs', 'bd'].edge_label
        edge_label_index = batch['tx', 'belongs', 'bd'].edge_label_index
        z = self.model(batch.x_dict, batch.edge_index_dict)
        output = self.model.decode(z, edge_label_index)
        probs = torch.sigmoid(output)

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
