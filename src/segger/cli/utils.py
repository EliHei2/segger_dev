import click
import yaml
from pydoc import locate
from argparse import Namespace
import typing
import os
import logging


def add_options(
    config_path: os.PathLike,
    show_default: bool = True,
):
    """
    A decorator to add command-line options to a Click command from a YAML
    configuration file.

    Parameters:
    config_path (os.PathLike): The path to the YAML configuration file
    containing the options.
    show_default (bool): Whether to show default values in help.

    Returns:
    function: A decorator function that adds the options to the Click command.

    The YAML configuration file should have the following format:
    ```
    option_name:
        type: "type_name"  # Optional, the type of the option
        (e.g., "str", "int")
        help: "description"  # Optional, the help text for the option
        default: value  # Optional, the default value for the option
        required: true/false  # Optional, whether the option is required
        ...
    ```
    Example usage:
    ```
    # config.yaml
    name:
        type: "str"
        help: "Your name"
        required: true
    age:
        type: "int"
        help: "Your age"
        default: 30

    # script.py
    @add_options('config.yaml')
    @click.command()
    def greet(args):
        click.echo(f"Hello, {args.name}! You are {args.age} years old.")
    ```
    """

    def decorator(function: typing.Callable):
        # Wrap the original function to convert kwargs to a Namespace object
        def wrapper(**kwargs):
            args_namespace = Namespace(**kwargs)
            return function(args_namespace)

        # Load the YAML configuration file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file.read())

        # Decorate function with all options
        for name, kwargs in reversed(config.items()):
            kwargs["show_default"] = show_default
            if "type" in kwargs:
                kwargs["type"] = locate(kwargs["type"])
            wrapper = click.option(f"--{name}", **kwargs)(wrapper)

        return wrapper

    return decorator


class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter to add color-coded log levels to the log messages.

    Attributes:
    grey (str): ANSI escape code for grey color.
    yellow (str): ANSI escape code for yellow color.
    red (str): ANSI escape code for red color.
    bold_red (str): ANSI escape code for bold red color.
    reset (str): ANSI escape code to reset color.
    format (str): The format string for log messages.
    FORMATS (dict): A dictionary mapping log levels to their respective
    color-coded format strings.

    Methods:
    format(record):
        Format the specified record as text, applying color codes based on the
        log level.
    """

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
