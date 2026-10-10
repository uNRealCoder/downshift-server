"""Typer option types that the commands share: the `Annotated` aliases and the enums that they
name. This module imports only the standard library, typer, and downshift.serve.options and
schemas. Both of these are free of torch. An import of this module therefore never pays the
import cost of torch, and `--help` and `--version` stay fast.
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
        help="A form that is already on this machine: model.onnx | weights.pt | "
        "downloaded-hf-repo-dir/ (has config.json) | pkg.module:attr. Hub ids are not accepted",
    ),
]
InputsOpt = Annotated[
    str | None,
    typer.Option(
        "--inputs",
        metavar="pkg.module:fn",
        help="Example inputs: a tuple, or a function that returns one",
    ),
]
ModelClassOpt = Annotated[
    str | None,
    typer.Option(
        "--model-class", metavar="pkg.module:Class", help="The class to load a state dict into"
    ),
]
UnsafeLoadOpt = Annotated[
    bool,
    typer.Option(
        "--unsafe-load", help="Allow torch.load(weights_only=False). This runs code from the file"
    ),
]
AdapterOpt = Annotated[
    str | None,
    typer.Option(
        "--adapter",
        metavar="NAME|path/to/adapter.py[:attr]",
        help="The model-family adapter: generic, pyg, hf, or your own adapter.py. "
        "Default: detect the adapter",
    ),
]
SamplesOpt = Annotated[
    int, typer.Option("-k", "--samples", min=1, help="The number of verification samples")
]
DynamicOpt = Annotated[
    str | None,
    typer.Option(
        "--dynamic",
        metavar="NAME:AXIS[,...]",
        help='The dynamic axes, for example "x:0,edge_index:1". Default: axis 0 of every input.',
    ),
]
AxisMaxOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--axis-max",
        metavar="NAME=N",
        help="Serve at most N along the named axis. You can repeat this option. The banner and "
        "/schema list the names. Generic models use dim0 or <input>_<axis> (--dynamic). Hugging "
        "Face models use batch and seq. PyG models use num_nodes and num_edges. Default: the "
        "bound that the export chose",
    ),
]
ExportCacheDirOpt = Annotated[
    str | None,
    typer.Option(
        "--export-cache-dir",
        metavar="DIR",
        help="An existing directory. Downshift saves the verified export there and reuses it "
        "at the next boot. The key is the content of the weights. Default: nothing is written "
        "to disk. To clear the cache, delete the directory",
    ),
]
ReferenceOpt = Annotated[
    str | None,
    typer.Option(
        "--reference",
        metavar="MODEL",
        help="A local PyTorch model to verify a .onnx file against. It accepts the same forms "
        "as MODEL",
    ),
]
TokenizerFromOpt = Annotated[
    str | None,
    typer.Option(
        "--tokenizer-from",
        metavar="DIR",
        help=(
            "A Hugging Face repo directory. Downshift loads the tokenizer, the pooling recipe "
            "and the label metadata from it, for a .onnx or PyTorch MODEL that has none of "
            "its own. This option does not depend on --reference. It never changes the "
            "numeric verification."
        ),
    ),
]
IntraOpThreadsOpt = Annotated[
    int,
    typer.Option(
        "--intra-op-threads",
        min=0,
        help=(
            "The number of threads inside one operation, for the backend that serves (ONNX "
            "Runtime or torch). 0 = the default of the backend. More threads decrease the "
            "latency of one request. They decrease throughput under concurrent load"
        ),
    ),
]
InterOpThreadsOpt = Annotated[
    int,
    typer.Option(
        "--inter-op-threads",
        min=0,
        help="The number of ONNX Runtime threads across operations. 0 = ONNX Runtime chooses",
    ),
]
WorkersOpt = Annotated[
    int,
    typer.Option(
        "--workers",
        min=1,
        help="The number of uvicorn worker processes. Each one loads, exports and warms up "
        "the model itself",
    ),
]
OutputEncodingOpt = Annotated[
    OutputEncoding,
    typer.Option(
        "--output-encoding",
        help="The default encoding of response tensors. A client can override it for one "
        "request with output_encoding",
    ),
]
MaxInputBytesOpt = Annotated[
    int,
    typer.Option(
        "--max-input-bytes",
        min=1,
        help="Reject a base64 tensor input that is larger than this value after decoding",
    ),
]
MaxBodyBytesOpt = Annotated[
    int,
    typer.Option(
        "--max-body-bytes",
        min=1,
        help="Reject a request body that is larger than this value. Downshift checks it "
        "before it parses the JSON",
    ),
]
MaxConcurrencyOpt = Annotated[
    int,
    typer.Option(
        "--max-concurrency",
        min=1,
        help="The number of inferences that can run at the same time in each worker process. "
        "One inference does not usually fill all cores, so several inferences at the same "
        "time serve more requests for each second. Each inference holds its own activation "
        "memory. If large requests run out of memory, decrease the value",
    ),
]
ExecutionOpt = Annotated[
    ExecutionChoice,
    typer.Option(
        "--execution",
        help="threadpool: worker threads parse, infer and encode each request. "
        "inline: the event loop runs small JSON bodies. Use inline only for models that take "
        "less than about 1 ms for one inference. A slower model stalls /health and /ready",
    ),
]
PrepThreadsOpt = Annotated[
    int,
    typer.Option(
        "--prep-threads",
        min=1,
        help="The number of threads in each worker process that parse and convert request "
        "bodies. They are separate from the inference threads (default: min(4, usable CPUs))",
    ),
]
MaxQueueOpt = Annotated[
    int,
    typer.Option(
        "--max-queue",
        min=0,
        help="The number of predicts that can wait beyond --max-concurrency. After that, a new "
        "predict gets a fast 503",
    ),
]
RequestTimeoutOpt = Annotated[
    float,
    typer.Option(
        "--request-timeout",
        min=0,
        help="The number of seconds that a predict can wait without a start. After this time, "
        "it gets a 503 and no inference. 0 = no limit",
    ),
]
AtolOpt = Annotated[
    float | None,
    typer.Option("--atol", help="Override the absolute tolerance. Default: by output dtype"),
]
RtolOpt = Annotated[
    float | None,
    typer.Option("--rtol", help="Override the relative tolerance. Default: by output dtype"),
]
SeedOpt = Annotated[
    int, typer.Option("--seed", help="The seed for the generation of verification samples")
]
VaryOpt = Annotated[
    str | None,
    typer.Option(
        "--vary",
        metavar="pkg.module:fn",
        help="fn(i) -> inputs for the verification samples. It replaces the sampler of the "
        "adapter. fn(0) must return the example inputs",
    ),
]
PoolingOpt = Annotated[
    PoolingChoice | None,
    typer.Option(
        "--pooling",
        help="For Hugging Face encoder repos. It overrides the pooling that the repo declares "
        "(modules.json). If the repo declares none, it sets one. The choices are mean, cls, "
        "max, mean_sqrt_len, lasttoken and weightedmean. 'none' serves token vectors",
    ),
]
NormalizeOpt = Annotated[
    bool | None,
    typer.Option(
        "--normalize/--no-normalize",
        help="For Hugging Face encoder repos. L2-normalise the embedding. Default: the setting "
        "of the repo",
    ),
]
JsonOpt = Annotated[
    bool, typer.Option("--json", help="Print the verdict as JSON on stdout. Logs go to stderr")
]
LogLevelOpt = Annotated[
    LogLevel,
    typer.Option(
        "--log-level",
        help="Diagnostics go to stdout through Python logging. The boot banner and the "
        "check and export reports always print. info adds one line for each request and the "
        "lines of uvicorn. debug adds tracebacks",
    ),
]
AccessLogOpt = Annotated[
    bool,
    typer.Option(
        "--access-log/--no-access-log",
        help="One log line for each request: method, path, status, duration and request ID",
    ),
]
