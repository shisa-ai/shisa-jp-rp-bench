"""One command-line interface for generation, judging, and score maintenance."""
from pathlib import Path

import click
import yaml

from .batch import main as batch
from .judging import main as judge
from .run import generate_conversations
from .scores import main as scores


@click.group()
def main():
    """Benchmark Japanese roleplay through an OpenAI-compatible API."""


@main.command()
@click.option('--config', required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--max-workers', type=click.IntRange(min=1))
@click.option('--max-samples', type=click.IntRange(min=1))
def generate(config, max_workers, max_samples):
    """Generate conversations from a YAML config and print their output path."""
    try:
        settings = yaml.safe_load(config.read_text(encoding='utf-8'))
        if not isinstance(settings, dict):
            raise ValueError('Configuration must be a mapping')
        for name, value in (('max_workers', max_workers), ('max_samples', max_samples)):
            if value is not None:
                settings[name] = value
        click.echo(str(generate_conversations(settings)))
    except yaml.YAMLError:
        raise click.ClickException('Invalid YAML configuration') from None
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


main.add_command(judge, 'judge')
main.add_command(batch, 'batch')
main.add_command(scores, 'scores')
