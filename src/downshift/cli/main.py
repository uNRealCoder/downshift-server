import typer
from downshift import __version__

app = typer.Typer(help="Universal PyTorch → ONNX verification and serving.")


@app.command()
def version():
    """Print the version."""
    typer.echo(f"downshift v{__version__}")


if __name__ == "__main__":
    app()
