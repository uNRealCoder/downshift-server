"""Smoke tests for CLI rendering: every table/banner branch gets executed at least once.

Text content is deliberately not asserted in detail (see tests/test_cli.py's docstring);
these tests just exercise the branches with constructed verdicts and serving states.
"""

from downshift.cli import render
from downshift.export.verdict import ExportVerdict
from downshift.export.verify import NumericsReport
from downshift.serve.backends import BackendMeta
from downshift.serve.engine import ServeOptions, ServingState
from downshift.serve.schemas import OutputEncoding


class _StubBackend:
    def __init__(self, name: str = "onnxruntime", device: str = "cpu") -> None:
        self.name = name
        self._device = device

    def metadata(self) -> BackendMeta:
        return BackendMeta(self.name, self._device, [], [])

    def infer(self, inputs):
        raise NotImplementedError


def _numerics(passed: bool, shape_generalization: bool = True) -> NumericsReport:
    return NumericsReport(
        samples_tested=8,
        max_abs_err=0.0 if passed else 1.0,
        max_rel_err=0.0 if passed else 1.0,
        failures=0 if passed else 3,
        shape_generalization=shape_generalization,
        tolerance_abs=1e-4,
        tolerance_rel=1e-3,
    )


def _verdict(**overrides) -> ExportVerdict:
    fields = dict(
        status="CLEAN",
        model_family="generic-torch",
        capture_strategy="strict=False",
        opset=18,
        op_types=["Gemm", "Relu"],
        numerics=_numerics(True),
        recommended_backend="onnxruntime",
        reason="exported via strict=False; numerics ok",
    )
    fields.update(overrides)
    return ExportVerdict(**fields)


def test_print_verdict_clean_with_dynamic_dims():
    verdict = _verdict(dynamic_dims={"x": [0]})
    render.print_verdict(verdict, "model")


def test_print_verdict_unverified_has_no_numerics():
    verdict = _verdict(status="UNVERIFIED", numerics=None, recommended_backend="onnxruntime")
    render.print_verdict(verdict, "model")


def test_print_verdict_degraded_numerics_failed():
    verdict = _verdict(
        status="DEGRADED",
        numerics=_numerics(False, shape_generalization=False),
        recommended_backend="torch",
    )
    render.print_verdict(verdict, "model")


def test_print_verdict_failed_shows_unsupported_ops_and_warnings():
    verdict = _verdict(
        status="FAILED",
        numerics=None,
        recommended_backend="torch",
        unsupported_ops=["scatter_reduce"],
        warnings=["tied weights: b shares storage with a"],
    )
    render.print_verdict(verdict, "model")


def test_print_artifacts_nothing_written(capsys):
    render.print_artifacts(None, None)
    assert "nothing written" in capsys.readouterr().out


def test_print_artifacts_writes_onnx_and_manifest(tmp_path, capsys):
    onnx_path = tmp_path / "m.onnx"
    manifest_path = tmp_path / "m.manifest.json"
    render.print_artifacts(onnx_path, manifest_path)
    out = capsys.readouterr().out
    assert "Wrote" in out
    assert "Manifest" in out


def test_print_artifacts_writes_onnx_without_manifest(tmp_path, capsys):
    render.print_artifacts(tmp_path / "m.onnx", None)
    out = capsys.readouterr().out
    assert "Wrote" in out
    assert "Manifest" not in out


def _serving_state(verdict: ExportVerdict, **overrides) -> ServingState:
    fields = dict(
        source="tests.models.clean_mlp:make_model",
        verdict=verdict,
        backend=_StubBackend(),
        input_names=("x",),
        options=ServeOptions(),
        notes=[],
    )
    fields.update(overrides)
    return ServingState(**fields)


def test_print_banner_unverified_without_reference():
    verdict = _verdict(status="UNVERIFIED", numerics=None, recommended_backend="onnxruntime")
    state = _serving_state(verdict)
    render.print_banner(state, "127.0.0.1", 8000)


def test_print_banner_shows_output_encoding(capsys):
    state = _serving_state(_verdict(), options=ServeOptions(output_encoding=OutputEncoding.base64))
    render.print_banner(state, "127.0.0.1", 8000)
    out = capsys.readouterr().out
    assert "Encoding" in out
    assert "base64" in out
    assert "output_encoding" in out
    assert "[fast]" not in out  # pybase64 is installed in the dev environment


def test_print_banner_tips_pybase64_when_missing(capsys, monkeypatch):
    monkeypatch.setattr(render, "BASE64_CODEC", "stdlib")
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    assert "downshift-server[fast]" in capsys.readouterr().out


def test_print_banner_unverified_with_prepared_backend_torch_skip():
    verdict = _verdict(
        status="UNVERIFIED",
        numerics=None,
        recommended_backend="torch",
        reason="--backend torch: export skipped",
        prepared=object(),
    )
    state = _serving_state(verdict, backend=_StubBackend("torch"), options=ServeOptions(backend="torch"))
    render.print_banner(state, "127.0.0.1", 8000)


def test_print_banner_failed_shows_reason():
    verdict = _verdict(
        status="FAILED", numerics=None, recommended_backend="torch", prepared=object()
    )
    state = _serving_state(verdict, backend=_StubBackend("torch"))
    render.print_banner(state, "127.0.0.1", 8000)


def test_print_banner_degraded_without_numerics_uses_reason():
    verdict = _verdict(
        status="DEGRADED",
        numerics=None,
        recommended_backend="torch",
        reason="exported but numerics diverge",
    )
    state = _serving_state(verdict, backend=_StubBackend("torch"))
    render.print_banner(state, "127.0.0.1", 8000)


def test_print_banner_degraded_forced_onnx_notes_and_warnings():
    verdict = _verdict(
        status="DEGRADED",
        numerics=_numerics(False, shape_generalization=False),
        recommended_backend="torch",
        dynamic_dims={"x": [0]},
        warnings=["model was in training mode; switched to eval() for export"],
    )
    options = ServeOptions(force_onnx=True)
    state = _serving_state(
        verdict,
        backend=_StubBackend("onnxruntime"),
        options=options,
        notes=[
            "--force-onnx: serving a DEGRADED graph; outputs may be wrong",
            "auto-selected backend note",
        ],
    )
    render.print_banner(state, "127.0.0.1", 8000)


def test_warn_and_error(capsys):
    render.warn("careful")
    render.error("bad thing")
    err = capsys.readouterr().err
    assert "careful" in err
    assert "bad thing" in err


def test_print_traceback(capsys):
    try:
        raise ValueError("boom")
    except ValueError:
        render.print_traceback()
    assert "boom" in capsys.readouterr().err
