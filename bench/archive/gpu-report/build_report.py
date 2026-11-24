import re

T = open("bench/archive/gpu-report/gpu_report_tables.md", encoding="utf-8").read()
tab = {}
for part in re.split(r"^@@", T, flags=re.M)[1:]:
    name, body = part.split("\n", 1)
    tab[name.strip()] = body.strip()

GPU = f"""## GPU run (NVIDIA RTX 4050 Laptop)

Same questions, on the GPU. Raw data: `bench/results/v0.6.0/torch_vs_ort_cuda.json`. Same harness with `--device cuda`.

### Answer for the GPU

**On this GPU, the model size decides, and ONNX Runtime is not always the faster backend.**

- **The GPU is worth using only for the compute-heavy models.** For `clean_mlp`, `dynamic_batch_cnn`, `gnn_gcn` and `tiny_bert` the best CPU result beats the best GPU result at batch 1 and 32 by 2x to 17x. The GPU wins on `mlp_large` (3.9x at batch 32 in exact fp32), `cnn_large` (3.7x at batch 32) and `bert_small` (2.2x at batch 1, 6.6x at batch 32, though that cell is noisy). See the CPU versus GPU table.
- **At batch 1, ORT is faster than eager torch on the GPU for every model: 1.2x to 1.9x.** The latency gap is small in absolute terms (`clean_mlp` p50 0.21 ms against 0.24 ms). `torch.compile` closes the gap on the two transformers (`tiny_bert` 588 against ORT 595; `bert_small` 334 against 317).
- **At large batch, the CPU result repeats: ORT leads on small and medium models and torch leads on the large encoder.** At batch 256 with data already on the GPU, ORT is 1.3x to 1.9x faster than eager on `clean_mlp`, `dynamic_batch_cnn`, `gnn_gcn` and `tiny_bert`. On `bert_small` torch is 1.4x faster in exact fp32 (5,157 against 3,622 items/s). With host copies included, `torch.compile` is also ahead there (4,920 against 3,456).
- **TF32 changes the answer, not only the speed.** The ORT CUDA provider turns TF32 on by default, and torch turns it off for matmul. With TF32 the output differs from exact fp32 by up to 2.2e-3 (against 2e-6 without it). It is worth 7% to 22% on the transformers in the cells that are not noisy. All the numbers below keep the two apart.
- **GPU use is low unless the model is big.** At batch 1 the GPU is 4% to 34% busy. Only `mlp_large`, `cnn_large` and `bert_small` at batch 256 reach 83% to 99%. In every GPU run one CPU core stays busy (0.87 to 1.02 cores), so the GPU does not free the CPU.
- **Memory.** A GPU process holds 440 to 1,270 MB of host RAM (CUDA libraries), against 60 to 481 MB in the CPU run. ORT used 2.2 GB of the 6 GB card on `bert_small` at batch 256, three times the 0.7 GB of torch.

### How the GPU run was set up

- **Hardware and software.** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, driver 610.88). A separate venv, `.venv-gpu`, with torch 2.14.0+cu130 and onnxruntime-gpu 1.30.0 (the same versions as the CPU run, apart from the CUDA builds). `torch.compile` on the GPU needs Triton, and Triton has no official Windows build. I used the community fork `triton-windows` 3.8.0.post29. Treat the `compile` columns as "compile with that fork".
- **Variants.** There is no TorchScript variant on the GPU. TF32 gets its own variants, because it is a speed trick that changes the answer.

| column | what it is |
|---|---|
| eager (fp32) | `nn.Module`, TF32 off for matmul and cuDNN |
| torch TF32 | the same, TF32 on |
| compile (fp32) | `torch.compile` (Inductor with Triton), TF32 off |
| ORT fp32 | ORT CUDA provider with `use_tf32=0` |
| ORT TF32 | ORT CUDA provider with its defaults. This is what `--device cuda` gives today. |

- **Two timing modes.** *Host I/O*: numpy in, numpy out, so the copies to and from the GPU are inside the timed call (a server pays this). *Resident*: the tensors are already on the GPU (torch) or bound with IO binding (ORT), so only the compute is timed.
- **Grid.** Batch 1, 32 and 256, one host thread. 240 runs in 62 minutes. `scatter_include_self_false` did not run on the GPU: its `forward` builds a tensor with `torch.zeros(...)` without `device=`, so torch stops with "Expected all tensors to be on the same device". That is a mistake in the test model. ORT is wrong on it in any case.
- **GPU metrics.** NVML sampled every 50 ms from the orchestrator while the worker ran: GPU utilization and power over the timed window, and peak device memory minus the memory before the worker started. A power of about 20 W appeared in most rows. I did not measure idle power separately, so I treat only the large readings (the `bert_small` rows at 70 to 105 W) as clear load.
- **Noise.** The median spread between the three rounds of a run is 11%. But 31 of the 210 comparable cells have rounds that differ by more than 50%, and some differ by 2x or more. They are all on `cnn_large`, `bert_small` and `mlp_large` at batch 32 and 256, so they are the cells that matter most. I marked them with †. I did not find the cause. A laptop GPU changes its clocks with power and heat, which is a suspect, but I did not test it. The conclusions above use only cells without †, except where I say so.

### CPU versus GPU

Best result over all variants, items per second, with host copies included for the GPU. The CPU column is the best of all CPU variants and both thread counts, from the CPU run above. I did not run the CPU at batch 256.

{tab["VSCPU"]}

The "TF32 allowed" columns take the best of all five GPU variants. The "fp32" columns leave out the TF32 variants, so their precision matches the CPU run. At batch 256 the GPU does 1.0M items/s on `clean_mlp`, still below the 2.1M of the CPU at batch 32.

### Throughput on the GPU

Items per second, higher is better. † marks a cell whose rounds differ by more than 50%.

**Batch 1, host I/O**

{tab["THR_host_b1"]}

**Batch 32, host I/O**

{tab["THR_host_b32"]}

**Batch 256, host I/O**

{tab["THR_host_b256"]}

**Batch 256, data already on the GPU (compute only)**

{tab["THR_resident_b256"]}

What the tables say:

1. **Batch 1 is limited by launch and copy cost, as on the CPU.** `clean_mlp` does 3.5k to 4.2k items/s on the GPU against 76.7k on the CPU. A GPU call is a few hundred microseconds, however small the model is. ORT leads eager by 1.2x to 1.9x here. I did not profile the kernels, so I do not know the reason.
2. **`torch.compile` helps where there are many small kernels.** At batch 1 it is 1.75x faster than eager on `tiny_bert` (588 against 337) and 2.0x on `bert_small` (334 against 168). On the small MLP and CNN models it is slower than eager (0.7x to 0.85x). At batch 256 with host copies it is 1.8x faster on `tiny_bert` and 1.45x on `gnn_gcn`. It costs 15 to 36 s at start-up, against 1 to 2 s for ORT.
3. **Small and medium models favour ORT at large batch.** In the resident table ORT fp32 is ahead of eager by 1.3x (`clean_mlp`), 1.7x (`dynamic_batch_cnn`), 1.6x (`gnn_gcn`) and 1.9x (`tiny_bert`). Only `torch.compile` gets near it on `tiny_bert` (157k against 137k).
4. **The large encoder favours torch at large batch.** `bert_small` at batch 256: eager 5,157 against ORT 3,622 (fp32), and 6,315 against 4,084 with TF32. This is the same direction as the CPU result, on a different processor and different kernels.
5. **Mode matters.** Host copies cost the most where the model is cheap. `clean_mlp` at batch 256 goes from 1.15M items/s resident to 0.81M with copies for eager torch.

### Resources on the GPU

Batch 256, data already on the GPU. Compared: eager torch and ORT fp32.

{tab["RES"]}

At batch 1 (host I/O) the GPU is barely used:

{tab["RES1"]}

1. **Small models do not use the GPU, and they do not need to.** At batch 256 the GPU is 6% to 48% busy on the four small models. The CPU does the same work faster (see the CPU versus GPU table).
2. **One CPU core is always busy.** The host thread launches kernels and waits for results. Moving to the GPU does not reduce CPU load per process, and it adds the CUDA libraries to host memory (440 to 1,270 MB).
3. **ORT keeps a large memory pool.** On `bert_small` at batch 256 ORT held 2,159 MB of GPU memory against 715 MB for torch. On a 6 GB card that limits the batch size and how many models can share the GPU. The pool is the same behaviour that made ORT use more memory than torch on the CPU at batch 32.
4. **Power.** Only the `bert_small` rows show a clear load (70 to 105 W). Torch drew 104 W and ORT 90 W for 5,157 and 3,622 items/s: about 20 and 25 mJ per item. Other rows sit near the 20 W floor and do not separate the variants.

### Precision: TF32 against exact fp32

Largest absolute difference from the eager CPU torch result, batch 256, data on the GPU.

{tab["ERR"]}

Both frameworks lose about the same precision with TF32 (up to 2.2e-3 on `gnn_gcn`, 5.6e-4 on `bert_small`). In exact fp32 both stay under 2.1e-6. ORT's `dynamic_batch_cnn` TF32 result shows no loss (I did not check why). This is a property of the GPU math mode, not of ORT or torch.

### Start-up on the GPU (batch 1, host I/O)

{tab["SETUP"]}

The eager column includes `import torch`, building the model and the first CUDA call, so it is not a fair load-time number. The ORT column is a session start with the CUDA provider.

### What the GPU result means for downshift

1. **`--device cuda` is not a general speed-up.** For small and mid-size models it is 2x to 17x slower than the CPU on this machine. It pays off from a 512-wide MLP at batch 32, a 32x32 convolution network at batch 32, and a small BERT at any batch. The docs could say that, with these crossover points as a rule of thumb.
2. **The default CUDA provider uses TF32.** `src/downshift/serve/backends.py` passes provider names without options, so ORT uses its default (TF32 on). The verifier defaults to `CPUExecutionProvider` (`src/downshift/core/verify.py:127`), and "TF32" appears nowhere in `src/` or `docs/`. The numbers above mean a model verified to 1e-6 on the CPU can differ by up to 2.2e-3 when it serves on the GPU. I did not check whether `serve --device cuda` verifies on the GPU. A candidate change is a `--precision` choice or a note in `docs/production.md`. I did not make any change.
3. **The verdict does not know about speed.** Same conclusion as the CPU run: `bert_small` at batch 256 is 1.4x faster on torch than on ORT, on both processors.
4. **Models that create CPU tensors in `forward` fail on the GPU** (the `scatter_include_self_false` test model). I did not run it through `downshift serve`.

"""

p = ".docs/REPORT_ONNX_VS_TORCH.md"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


sub(
    "Measured 2026-10-24 on one laptop CPU. Raw data: `bench/results/v0.6.0/torch_vs_ort.json`. Harness: `bench/torch_vs_ort.py`.",
    "Measured 2026-10-24 on one laptop, on the CPU and on the GPU. Harness: `bench/torch_vs_ort.py`. Raw data: `bench/results/v0.6.0/torch_vs_ort.json` (CPU) and `torch_vs_ort_cuda.json` (GPU). The first sections cover the CPU. The GPU run starts at \"GPU run\".",
)
sub("## Answer\n", "## Answer for the CPU\n")
sub(
    'That is a good argument for the verdict-driven design and against a fixed "always ONNX" policy.\n',
    'That is a good argument for the verdict-driven design and against a fixed "always ONNX" policy.\n\n'
    "**The GPU run (end of this report) points the same way:** ORT is faster at batch 1 and on small and medium models, torch is faster on the large encoder at large batch, and the GPU itself is worth it only for the compute-heavy models.\n",
)
sub(
    "Windows 11, CPU only. Python 3.12, torch 2.14.0+cpu, onnxruntime 1.30.0.",
    "Windows 11. CPU run: Python 3.12, torch 2.14.0+cpu, onnxruntime 1.30.0.",
)
sub(
    "- **One machine, CPU only.** An 8-core laptop CPU (AMD Zen 4), Windows. No GPU, no Linux, no Intel CPU. ORT's CPU kernels",
    "- **One machine.** An 8-core laptop CPU (AMD Zen 4) and an RTX 4050 Laptop GPU, Windows. No Linux, no Intel CPU, no desktop or data-centre GPU. ORT's CPU kernels",
)
sub(
    "- **No quantization and no reduced precision.** All float32. ORT int8 and torch bf16 paths can change the result for large models.",
    "- **No quantization and no reduced precision.** All float32 (plus TF32 on the GPU). Half precision (fp16 or bf16) is the usual GPU speed-up and I did not test it. ORT int8 and torch bf16 paths can change the result for large models.",
)
sub("## Limits of this test\n", GPU + "## Limits of this test\n")
sub(
    "- **Memory is the process peak.**",
    "- **GPU limits.** One laptop GPU, whose clocks and power change with heat (suspected cause of the noisy cells, not tested). Host I/O uses ordinary (pageable) memory, no pinned buffers and no CUDA graphs. ORT's TensorRT provider is listed as available and I did not test it. `torch.compile` ran on the `triton-windows` fork. One host thread. The `scatter_include_self_false` model did not run on the GPU.\n- **Memory is the process peak.**",
)
sub(
    "python -m bench.torch_vs_ort --case bert_small --seconds 3   # one model, longer windows\n```",
    "python -m bench.torch_vs_ort --case bert_small --seconds 3   # one model, longer windows\n\n"
    "# GPU: needs a venv with a CUDA build of torch, onnxruntime-gpu and nvidia-ml-py.\n"
    "# onnxruntime-gpu and torch must use the same CUDA major version (here CUDA 13).\n"
    "python -m bench.torch_vs_ort --device cuda    # 240 runs, about 60 minutes\n```",
)
sub(
    "The output file is `bench/results/v0.6.0/torch_vs_ort.json`.",
    "The output files are `bench/results/v0.6.0/torch_vs_ort.json` (CPU) and `torch_vs_ort_cuda.json` (GPU).",
)
open(p, "w", encoding="utf-8").write(s)
print("ok", len(s.splitlines()), "lines")
