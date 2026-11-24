"""The data types of downshift: every dataclass, and the enums and aliases that they need.

This package has no logic of its own. A module here imports only what its types need, and it
keeps torch under TYPE_CHECKING, so the CLI can import most of it at no cost. The exception is
`serving.py`: `ServingState` needs the backends and the result cache.

Each module groups the types of one job:

  phase        the boot phases, and what /ready reports
  options      ServeOptions and the choices of the serve options
  axes         the dynamic axes of a model: bounds and facts
  adapters     what an adapter returns, and the Hugging Face text types
  verdict      ExportVerdict and the numbers behind it
  loading      LoadSpec and LoadedModel: how to load a model, and what was loaded
  cache        the export cache entry and the rows of the result cache
  requests     a decoded predict request and the plan of a cached request
  signatures   the function types of the request path
  backend      the input and output description of a backend
  serving      ServingState: everything that one server process holds
  metrics      the counters behind GET /metrics
  cli          the data that `serve --workers N` sends to each worker
"""
