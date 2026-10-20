"""The row check: does a row give the same output alone and in a batch with other rows?

The result cache serves rows on their own, so it needs this. `x - x.mean(0)` fails it, and so
does an output whose sequence axis follows the longest row.
"""

from collections.abc import Callable

import numpy as np
import torch

from downshift.adapters.base import Family, Prepared
from downshift.adapters.embedding import PADDING_SIDE_ATTR
from downshift.core.verify import NumericsReport, as_tensor_list, make_shared_axis0_vary_fn

# Runs a tuple of input tensors, gives one numpy array for each output.
Runner = Callable[[tuple], list[np.ndarray]]


def batch_axis(prepared: Prepared, numerics: NumericsReport) -> str | None:
    """None if all inputs share one dynamic axis-0 Dim, and axis 0 of each output follows
    it while no other output axis changes. Otherwise the reason."""
    names = set()
    for name, spec in zip(prepared.input_names, prepared.dynamic_shapes, strict=True):
        dim = spec.get(0) if spec else None
        if dim is None:
            return f"input {name} has no dynamic batch axis"
        names.add(getattr(dim, "__name__", str(dim)))
    if len(names) > 1:
        return f"the inputs do not share one batch axis ({', '.join(sorted(names))})"

    first_inputs = numerics.sample_shapes
    outputs = numerics.output_shapes
    if not outputs or not first_inputs:
        return "no output shapes were measured"
    for idx in range(len(outputs[0])):
        for sample_inputs, sample_outputs in zip(first_inputs, outputs, strict=True):
            shape = sample_outputs[idx]
            if not shape or shape[0] != sample_inputs[0][0]:
                return f"output_{idx} axis 0 does not follow the batch size"
        for axis in range(1, len(outputs[0][idx])):
            if len({sample[idx][axis] for sample in outputs}) > 1:
                return (
                    f"output_{idx} axis {axis} changes with the input shape, "
                    "so a row cannot be cut from the batch output"
                )
    return None


def _row(sample: tuple, i: int) -> tuple:
    return tuple(t[i : i + 1] for t in sample)


def _text_rows(prepared: Prepared, first: tuple, second: tuple) -> tuple[tuple, list[tuple]] | None:
    """Two unpadded rows of different lengths, and their batch padded on the tokenizer's side."""
    if prepared.input_names != ("input_ids", "attention_mask"):
        return None
    side = getattr(getattr(prepared.model, "model", None), PADDING_SIDE_ATTR, "right")
    rows = []
    for ids, mask in (_row(first, 0), _row(second, 0)):
        keep = mask[0].bool()
        rows.append((ids[:, keep], mask[:, keep]))
    if rows[0][0].shape[1] == rows[1][0].shape[1] > 1:
        # Equal lengths test no padding. Cut one token from the second row.
        cut = rows[1][0].shape[1] - 1
        start = 0 if side == "right" else 1
        rows[1] = (rows[1][0][:, start : start + cut], rows[1][1][:, start : start + cut])
    width = max(r[0].shape[1] for r in rows)
    joined_ids, joined_mask = [], []
    for ids, mask in rows:
        pad = width - ids.shape[1]
        spec = (pad, 0) if side == "left" else (0, pad)
        joined_ids.append(torch.nn.functional.pad(ids, spec, value=0))
        joined_mask.append(torch.nn.functional.pad(mask, spec, value=0))
    joined = (torch.cat(joined_ids), torch.cat(joined_mask))
    return joined, [rows[0], rows[1]]


def two_row_sample(prepared: Prepared, seed: int = 0) -> tuple[tuple, list[tuple]] | str:
    """`(joined, [row_a, row_b])`: two rows as batches of one and together, or the reason
    why downshift cannot make them."""
    vary = prepared.vary_fn or make_shared_axis0_vary_fn(prepared.inputs, prepared.dynamic_shapes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        first, second = vary(0), vary(1)
    if prepared.family == Family.hf:
        pair = _text_rows(prepared, first, second)
        return pair if pair is not None else "the text inputs are not input_ids and attention_mask"
    row_a, row_b = _row(first, 0), _row(second, 0)
    if any(a.shape[1:] != b.shape[1:] for a, b in zip(row_a, row_b, strict=True)):
        return "no two samples with the same row shape"
    return tuple(torch.cat([a, b]) for a, b in zip(row_a, row_b, strict=True)), [row_a, row_b]


def rows_agree(
    run: Runner, joined: tuple, rows: list[tuple], atol: float, rtol: float
) -> str | None:
    """None if each row of the joined output is close to that row run alone. Otherwise the
    first difference, as text."""
    together = run(joined)
    for i, row in enumerate(rows):
        alone = run(row)
        if len(alone) != len(together):
            return "the number of outputs changes with the batch size"
        for idx, (a, t) in enumerate(zip(alone, together, strict=True)):
            got = np.asarray(t[i], dtype=np.float64)
            want = np.asarray(a[0], dtype=np.float64)
            if got.shape != want.shape or not np.allclose(
                got, want, rtol=rtol, atol=atol, equal_nan=False
            ):
                size = float(np.abs(got - want).max()) if got.shape == want.shape else float("nan")
                return f"output {idx} row {i} differs by {size:.1e} when it runs with other rows"
    return None


def torch_runner(model: torch.nn.Module) -> Runner:
    def run(feeds: tuple) -> list[np.ndarray]:
        with torch.inference_mode():
            outs = as_tensor_list(model(*feeds))
        return [o.detach().float().numpy() if o.is_floating_point() else o.numpy() for o in outs]

    return run


def session_runner(session, input_names: tuple[str, ...]) -> Runner:
    def run(feeds: tuple) -> list[np.ndarray]:
        outs: list[np.ndarray] = session.run(
            None, {n: t.numpy() for n, t in zip(input_names, feeds, strict=True)}
        )
        return outs

    return run


def check_rows(
    prepared: Prepared, numerics: NumericsReport, session
) -> tuple[bool | None, str | None]:
    """`(row_independent, reason)`: True if two rows agree alone and joined on both backends,
    False if not, None if downshift could not check. Two rows cannot prove that a small
    effect is absent."""
    if not numerics.passed:
        return None, "not checked: the numerics check failed"
    try:
        reason = batch_axis(prepared, numerics)
        if reason is not None:
            return False, reason
        sample = two_row_sample(prepared)
        if isinstance(sample, str):
            return None, f"not checked: {sample}"
        joined, rows = sample
        atol, rtol = numerics.tolerance_abs, numerics.tolerance_rel
        for run in (
            session_runner(session, prepared.input_names),
            torch_runner(prepared.model),
        ):
            reason = rows_agree(run, joined, rows, atol, rtol)
            if reason is not None:
                return False, reason
    except Exception as exc:  # noqa: BLE001 - a fact of the verdict, not a crash
        return None, f"not checked: {type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}"
    return True, None
