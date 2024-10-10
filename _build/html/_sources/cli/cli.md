# Command-Line Interface (CLI)

Documentation for the Segger CLI.

```python
import click

@click.group()
def cli():
    pass

@cli.command()
def create_dataset():
    """Create a new dataset."""
    pass

if __name__ == '__main__':
    cli()
