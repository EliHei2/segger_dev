import click
import typing
import os
from segger.cli.utils import add_options, CustomFormatter
from pathlib import Path
import logging


predict_yml = Path(__file__).parent / 'configs' / 'predict' / 'default.yaml'


@click.command(name="predict", help="Predict using the Segger model")
#@click.option('--foo', default="bar")  # add more options above, not below
@add_options(config_path=predict_yml)
def predict(args):

    # Setup
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[ch])

    # Import packages
    logging.info("Importing packages...")
    from segger.data.utils import XeniumDataset
    from torch_geometric.loader import DataLoader
    from segger.prediction.predict import load_model, predict
    logging.info("Done.")

    # Load datasets and model
    logging.info("Loading Xenium datasets and Segger model...")
    dataset = XeniumDataset(args.dataset_path)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
    )
    if len(data_loader) == 0:
        msg = f"Nothing to predict: No data found at '{args.dataset_path}'."
        logging.warning(msg)
        return
    lit_segger = load_model(args.checkpoint_path)
    logging.info("Done.")

    # Make prediction on dataset
    logging.info("Making predictions on data")
    predictions = predict(
        lit_segger=lit_segger,
        data_loader=data_loader,
        score_cut=args.score_cut,
        use_cc=args.use_cc,
    )
    logging.info("Done.")

    # Write predictions to file
    logging.info("Saving predictions to file")
    predictions.to_csv(args.output_path, index=False)
    logging.info("Done.")
