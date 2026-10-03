"""Typer option types shared across commands: the `Annotated` aliases and the enums they
name. Stdlib, typer and downshift.serve.options/schemas only (both torch-free), so importing
this module never pays torch's import cost; `--help` and `--version` stay fast.
"""

from enum import StrEnum
from typing import Annotated

import typer

from downshift.adapters.pooling import PoolingChoice
from downshift.serve.options import ExecutionChoice
from downshift.serve.schemas import OutputEncoding


class LogLevel(StrEnum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"


ModelArg = Annotated[
    str,
    typer.Argument(
        metavar="MODEL",
        help="Already on this machine: model.onnx | weights.pt | downloaded-hf-repo-dir/ "
        "(has config.json) | pkg.module:attr. Hub ids are not accepted",
    ),
]
InputsOpt = Annotated[
    str | None,
    typer.Option(
        "--inputs", metavar="pkg.module:fn", help="Example inputs: a tuple, or a factory for one"
    ),
]
ModelClassOpt = Annotated[
    str | None,
    typer.Option(
        "--model-class", metavar="pkg.module:Class", help="Class to load a state dict into"
    ),
]
UnsafeLoadOpt = Annotated[
    bool,
    typer.Option(
        "--unsafe-load", help="Allow torch.load(weights_only=False); runs code from the file"
    ),
]
AdapterOpt = Annotated[
    str | None,
    typer.Option(
        "--adapter",
        metavar="NAME|path/to/adapter.py[:attr]",
        help="Model-family adapter: generic, pyg, hf, or your own adapter.py; default: detect",
    ),
]
SamplesOpt = Annotated[
    int, typer.Option("-k", "--samples", min=1, help="Number of verification samples")
]
DynamicOpt = Annotated[
    str | None,
    typer.Option(
        "--dynamic",
        metavar="NAME:AXIS[,...]",
        help='Dynamic axes, e.g. "x:0,edge_index:1". Default: axis 0 of every input.',
    ),
]
AxisMaxOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--axis-max",
        metavar="NAME=N",
        help="Serve at most N along the named axis; repeatable. The banner and /schema list the "
        "names: dim0 or <input>_<axis> (--dynamic) for generic models, batch and seq for Hugging "
        "Face, num_nodes and num_edges for PyG. Default: the bound the export chose",
    ),
]
ExportCacheDirOpt = Annotated[
    str | None,
    typer.Option(
        "--export-cache-dir",
        metavar="DIR",
        help="An existing directory to save the verified export in and reuse on the next boot, "
        "keyed by the weights' content. Default: nothing is written to disk. Delete the "
        "directory to clear it",
    ),
]
ReferenceOpt = Annotated[
    str | None,
    typer.Option(
        "--reference",
        metavar="MODEL",
        help="Local PyTorch model to verify a .onnx against; same accepted forms as MODEL",
    ),
]
TokenizerFromOpt = Annotated[
    str | None,
    typer.Option(
        "--tokenizer-from",
        metavar="DIR",
        help=(
            "Hugging Face repo directory to load the tokenizer, pooling recipe and label "
            "metadata from, for a .onnx or PyTorch MODEL that has none of its own. "
            "Independent of --reference: this never affects numeric verification."
        ),
    ),
]
IntraOpThreadsOpt = Annotated[
    int,
    typer.Option(
        "--intra-op-threads", min=0, help="ORT threads within one op; 0 = let ONNX Runtime choose"
    ),
]
InterOpThreadsOpt = Annotated[
    int,
    typer.Option(
        "--inter-op-threads", min=0, help="ORT threads across ops; 0 = let ONNX Runtime choose"
    ),
]
WorkersOpt = Annotated[
    int,
    typer.Option(
        "--workers",
        min=1,
        help="Uvicorn worker processes; each independently loads/exports/warms the model",
    ),
]
OutputEncodingOpt = Annotated[
    OutputEncoding,
    typer.Option(
        "--output-encoding",
        help="Default encoding of response tensors; clients override per request with "
        "output_encoding",
    ),
]
MaxInputBytesOpt = Annotated[
    int,
    typer.Option(
        "--max-input-bytes",
        min=1,
        help="Reject base64 tensor inputs larger than this once decoded",
    ),
]
MaxBodyBytesOpt = Annotated[
    int,
    typer.Option(
        "--max-body-bytes",
        min=1,
        help="Reject request bodies larger than this, before they are parsed as JSON",
    ),
]
MaxConcurrencyOpt = Annotated[
    int,
    typer.Option(
        "--max-concurrency",
        min=1,
        help="Inferences allowed to run at once per worker process (default 1). Small models "
        "often serve more requests per second at 2-4, since one inference does not fill every "
        "core; each extra one holds its own activation memory",
    ),
]
ExecutionOpt = Annotated[
    ExecutionChoice,
    typer.Option(
        "--execution",
        help="threadpool: every request's parse, inference and encode run on worker threads. "
        "inline: small JSON bodies run on the event loop; only for models under ~1 ms per "
        "inference, a slower one stalls /health and /ready",
    ),
]
PrepThreadsOpt = Annotated[
    int,
    typer.Option(
        "--prep-threads",
        min=1,
        help="Threads per worker process that decode request bodies and encode responses, "
        "apart from the inference threads (default: min(4, usable CPUs))",
    ),
]
MaxQueueOpt = Annotated[
    int,
    typer.Option(
        "--max-queue",
        min=0,
        help="Predicts allowed to wait past --max-concurrency before a new one gets a fast 503",
    ),
]
RequestTimeoutOpt = Annotated[
    float,
    typer.Option(
        "--request-timeout",
        min=0,
        help="Seconds a predict may wait, unstarted, before a 503 instead of an inference; "
        "0 = no limit",
    ),
]
AtolOpt = Annotated[
    float | None,
    typer.Option("--atol", help="Absolute tolerance override; default: by output dtype"),
]
RtolOpt = Annotated[
    float | None,
    typer.Option("--rtol", help="Relative tolerance override; default: by output dtype"),
]
SeedOpt = Annotated[int, typer.Option("--seed", help="Seed for verification sample generation")]
VaryOpt = Annotated[
    str | None,
    typer.Option(
        "--vary",
        metavar="pkg.module:fn",
        help="fn(i) -> inputs for verification samples, overriding the adapter's own; "
        "fn(0) must return the example inputs",
    ),
]
PoolingOpt = Annotated[
    PoolingChoice | None,
    typer.Option(
        "--pooling",
        help="Hugging Face encoder repos: override the pooling the repo declares "
        "(modules.json), or set one when it declares none: mean, cls, max, mean_sqrt_len, "
        "lasttoken, weightedmean. 'none' serves token vectors",
    ),
]
NormalizeOpt = Annotated[
    bool | None,
    typer.Option(
        "--normalize/--no-normalize",
        help="Hugging Face encoder repos: L2-normalise the embedding; default is what the repo says",
    ),
]
JsonOpt = Annotated[
    bool, typer.Option("--json", help="Print the verdict as JSON on stdout; logs go to stderr")
]
LogLevelOpt = Annotated[
    LogLevel,
    typer.Option(
        "--log-level",
        help="Diagnostics go to stdout through Python logging. The boot banner and the "
        "check/export reports always print. info adds one line per request and uvicorn's own "
        "lines; debug adds tracebacks",
    ),
]
AccessLogOpt = Annotated[
    bool,
    typer.Option(
        "--access-log/--no-access-log",
        help="One log line per request: method, path, status, duration and request id",
    ),
]
