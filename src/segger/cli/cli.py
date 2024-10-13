from segger.cli.create_dataset import create_dataset
from segger.cli.train_model import train
from segger.cli.predict import predict
import click


# Setup main CLI command
@click.group(help="Command line interface for the Segger segmentation package")
def segger():
    pass


# Add sub-commands to main CLI commands
segger.add_command(create_dataset)
segger.add_command(train)
segger.add_command(predict)
