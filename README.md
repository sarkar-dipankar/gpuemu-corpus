# gpuemu-corpus

The 26-op kernel-correctness corpus used to measure P1–P4 in the
[gpuemu research artefact](https://github.com/sarkar-dipankar/gpuemu-paper),
packaged for `pip install` so the [gpuemu](https://github.com/Skelf-Research/gpuemu)
daemon can validate any project's kernels against the same controlled set.

## What's in it

- **16 correct controls** — 13 Triton kernels: `softmax_triton`,
  `gelu_triton`, `silu_triton`, `rmsnorm_triton`, `l2norm_triton`,
  `leaky_relu_triton`, `relu_triton`, `sigmoid_triton`, `tanh_triton`,
  `elu_triton`, `matmul_triton`, `attention_triton`,
  `flash_attention_triton`. Plus 3 numpy baselines: `softmax`,
  `layernorm`, `matmul`.
- **10 LLM-style buggy variants** — `softmax_llm_buggy`,
  `softmax_triton_buggy`, `gelu_triton_buggy`, `silu_triton_buggy`,
  `rmsnorm_triton_buggy`, `l2norm_triton_buggy`,
  `leaky_relu_triton_buggy`, `matmul_triton_buggy` (`acc=` instead of
  `acc+=`), `attention_triton_buggy` (missing `1/√D`),
  `flash_attention_triton_buggy` (missing `acc*α` rescale on max
  update).

Each op has:
- `meta.json` — name, source, benchmark_verdict, input_names, reference path,
  kernel path, dtypes, per-op tolerances, op_schema (shape generators).
- `ref_fp64.py` — high-precision fp64 reference invoked by the gpuemu daemon.
- `kernel.py` — the kernel under test (Triton for the buggy variants).

## Install

`gpuemu-corpus` is currently installable from source. PyPI publish is
planned for a follow-up release.

```bash
pip install git+https://github.com/sarkar-dipankar/gpuemu-corpus.git
```

Or, for local development with a checkout of `gpuemu-paper` alongside:

```bash
git clone https://github.com/sarkar-dipankar/gpuemu-corpus.git
cd gpuemu-corpus
python3 scripts/vendor_corpus.py --paper-repo ../gpuemu-paper
pip install -e .
```

## Register the corpus with the gpuemu daemon

```python
from pathlib import Path
from gpuemu_corpus import register_all

# Writes a gpuemu.toml registering every bundled op with absolute paths.
toml = register_all(client=None, toml_path=Path("gpuemu.toml"))
print(f"wrote {toml}")
```

Then start the daemon against the generated config:

```bash
gpuemu daemon start --config gpuemu.toml
gpuemu ci --parallel 8 --format sarif --output gpuemu.sarif
```

## Where it came from

The corpus was developed for the four-paper research program
([P1](https://github.com/sarkar-dipankar/gpuemu-paper/blob/main/papers/p1/paper.pdf):
correctness illusion in LLM-generated kernels,
[P2](https://github.com/sarkar-dipankar/gpuemu-paper/blob/main/papers/p2/paper.pdf):
tolerance calibration,
[P3](https://github.com/sarkar-dipankar/gpuemu-paper/blob/main/papers/p3/paper.pdf):
test-input generation strategies,
[P4](https://github.com/sarkar-dipankar/gpuemu-paper/blob/main/papers/p4/paper.pdf):
static PTX gating). The canonical source is `gpuemu-paper/corpus/`; this
package vendors a release-tagged snapshot at build time via
`scripts/vendor_corpus.py`.

## Building from source

```bash
git clone https://github.com/sarkar-dipankar/gpuemu-corpus
cd gpuemu-corpus
# Vendor the corpus from a local gpuemu-paper checkout.
python3 scripts/vendor_corpus.py --paper-repo ../gpuemu-paper
# Build the wheel.
python3 -m build
```

## License

Dual-licensed under MIT OR Apache-2.0.
