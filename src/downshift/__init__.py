"""downshift: check whether a PyTorch model survives ONNX export, then serve it."""

from pathlib import Path

from downshift.adapters.base import Adapter, Prepared
from downshift.export.manifest import write_manifest
from downshift.export.prevalidated import intake
from downshift.export.verdict import ExportVerdict, build_verdict, check, prepare_model
from downshift.export.verify import NumericsReport

__version__ = "0.2.0"


def export(
    model,
    output: str | Path,
    example_inputs: tuple | None = None,
    k: int = 8,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    fp16: bool = False,
    source_path: Path | None = None,
    verify_numerics: bool = True,
) -> ExportVerdict:
    """check() plus writing the .onnx and its manifest. `output` is the .onnx path.

    A FAILED verdict writes nothing; a DEGRADED one still writes the artifact because
    the manifest records exactly how far off it is.
    """
    verdict = check(
        model,
        example_inputs,
        k=k,
        adapter=adapter,
        dynamic=dynamic,
        fp16=fp16,
        verify_numerics=verify_numerics,
    )
    if verdict.onnx_program is None:
        return verdict
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    verdict.onnx_program.save(str(output))  # type: ignore[attr-defined]
    verdict.onnx_path = output
    write_manifest(output, verdict, source_path, __version__)
    return verdict


__all__ = [
    "Adapter",
    "ExportVerdict",
    "NumericsReport",
    "Prepared",
    "build_verdict",
    "check",
    "export",
    "intake",
    "prepare_model",
    "__version__",
]
