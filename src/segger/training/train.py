import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.typing import Metadata
from torchmetrics import AUROC, F1Score
from pytorch_lightning import LightningModule
from torch_geometric.nn import to_hetero
from segger.models.segger_model import Segger
from segger.data.utils import SpatialTranscriptomicsDataset
from typing import Any, Union, Tuple
import inspect


class LitSegger(LightningModule):
    """
    LitSegger is a PyTorch Lightning module for training and validating the Segger model.

    This class handles the training loop, validation loop, and optimizer configuration
    for the Segger model, which is designed for heterogeneous graph data. The model is
    compatible with large-scale datasets and leverages PyTorch Geometric for graph-related operations.

    Attributes
    ----------
    model : Segger
        The Segger model wrapped with PyTorch Geometric's `to_hetero` for heterogeneous graph support.
    validation_step_outputs : list
        A list to store outputs from the validation steps.
    criterion : torch.nn.Module
        The loss function used for training, specifically `BCEWithLogitsLoss`.

    Methods
    -------
    forward(batch: SpatialTranscriptomicsDataset) -> torch.Tensor
        Forward pass for the batch of data.
    training_step(batch: Any, batch_idx: int) -> torch.Tensor
        Defines a single training step.
    validation_step(batch: Any, batch_idx: int) -> torch.Tensor
        Defines a single validation step.
    configure_optimizers() -> torch.optim.Optimizer
        Configures the optimizer and learning rate scheduler.
    """
    
    def __init__(self, **kwargs):
        """
        Initializes the LitSegger module with the given parameters.

        This constructor determines whether to initialize the module with new parameters
        or with pre-defined components (e.g., for testing purposes).

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for initializing the module. The specific parameters depend
            on whether the module is being initialized with new parameters or with existing
            components.

            For new parameters:
                init_emb : int
                    The initial embedding size.
                hidden_channels : int
                    The number of hidden channels.
                out_channels : int
                    The number of output channels.
                heads : int
                    The number of attention heads.
                aggr : str
                    The aggregation method.
                metadata : Union[Tuple, Metadata]
                    Metadata for the heterogeneous graph.

            For components:
                model : Segger
                    The Segger model to be used.
        """
        super().__init__()

        new_args = inspect.getfullargspec(self.from_new)[0][1:]
        cmp_args = inspect.getfullargspec(self.from_components)[0][1:]

        # Normal constructor
        if set(kwargs.keys()) == set(new_args):
            self.from_new(**kwargs)

        # Component constructor for testing
        elif set(kwargs.keys()) == set(cmp_args):
            self.from_components(**kwargs)

        # Otherwise, throw an error
        else:
            raise ValueError(
                "Supplied kwargs do not match either constructor. Should be "
                f"one of '{new_args}' or '{cmp_args}'."
            )

        # Set up the loss function and validation output storage
        self.validation_step_outputs = []
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def from_new(self, init_emb: int, hidden_channels: int, out_channels: int, heads: int, aggr: str, metadata: Union[Tuple, Metadata]):
        """
        Initializes the LitSegger module with new parameters.

        Parameters
        ----------
        init_emb : int
            Initial embedding size.
        hidden_channels : int
            Number of hidden channels.
        out_channels : int
            Number of output channels.
        heads : int
            Number of attention heads in GAT layers.
        aggr : str
            Aggregation method for the `to_hetero` conversion.
        metadata : Union[Tuple, Metadata]
            Metadata for the heterogeneous graph, used to define the node and edge types.

        Notes
        -----
        This method creates a new Segger model with the specified parameters and
        wraps it in a `to_hetero` call to handle heterogeneous graph structures.
        """
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

    def from_components(self, model: Segger):
        """
        Initializes the LitSegger module with existing components for testing.

        Parameters
        ----------
        model : Segger
            The Segger model to be used.

        Notes
        -----
        This method directly assigns the provided Segger model to the LitSegger
        module, which can be useful for testing or reusing pre-trained models.
        """
        self.model = model

    def forward(self, batch: SpatialTranscriptomicsDataset) -> torch.Tensor:
        """
        Forward pass for the batch of data.

        This method computes the forward pass of the model for a batch of data, producing
        output tensors based on the input features and edge indices.

        Parameters
        ----------
        batch : SpatialTranscriptomicsDataset
            Batch of data, including node features and edge indices.

        Returns
        -------
        torch.Tensor
            The output tensor resulting from the forward pass.
        """
        z = self.model(batch.transcripts, batch.boundaries)
        output = torch.matmul(z['tx'], z['nc'].t())  # Example for bipartite graph output
        return output
        
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines the training step.

        This method computes the loss for a single batch during training, logs the
        loss value, and returns it for backpropagation.

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
        z = self.model(batch.transcripts, batch.boundaries)
        output = torch.matmul(z['tx'], z['nc'].t()) 
        edge_label_index = batch['tx', 'belongs', 'nc'].edge_label_index
        out_values = output[edge_label_index[0], edge_label_index[1]]
        edge_label = batch['tx', 'belongs', 'nc'].edge_label
        loss = self.criterion(out_values, edge_label)
        
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.transcripts.shape[0])
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Defines the validation step.

        This method computes the loss for a single batch during validation, logs the
        loss and other metrics (AUROC, F1 score), and returns the loss.

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
        z = self.model(batch.transcripts, batch.boundaries)
        output = torch.matmul(z['tx'], z['nc'].t()) 
        edge_label_index = batch['tx', 'belongs', 'nc'].edge_label_index
        out_values = output[edge_label_index[0], edge_label_index[1]]
        edge_label = batch['tx', 'belongs', 'nc'].edge_label
        loss = self.criterion(out_values, edge_label)

        # Compute metrics
        out_values = torch.sigmoid(out_values)
        auroc = AUROC(task="binary")
        auroc_res = auroc(out_values, edge_label)
        f1 = F1Score(task='binary').to(self.device)
        f1_res = f1(out_values, edge_label)
        
        # Log metrics
        self.log("validation_loss", loss, batch_size=batch.transcripts.shape[0])
        self.log("validation_auroc", auroc_res, prog_bar=True, batch_size=batch.transcripts.shape[0])
        self.log("validation_f1", f1_res, prog_bar=True, batch_size=batch.transcripts.shape[0])

        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        This method defines and returns the optimizer to be used during training,
        along with any learning rate schedulers if needed.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer used for training the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
