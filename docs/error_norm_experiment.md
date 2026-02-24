# Error-Norm vs Token-Index Experiment

This document describes how to run the **Figure 2–style** error-norm analysis
for `Qwen2.5-VL-3B-Instruct` on a single MMMU-Pro Vision sample.

---

## What the experiment does

The script `experiments/error_norm_analysis.py` runs two forward passes on the
same multimodal input (an MMMU-Pro Vision image + multiple-choice prompt) and
measures how the hidden-state error evolves across **token positions** and
**transformer layers**:

| Pass | Description |
|------|-------------|
| **recompute** | Standard forward pass with correct 3-D rotary position IDs (ground truth). |
| **reuse** | Same input but image-token temporal position IDs are reset to `[0, N_img)`, simulating KV-cache reuse from a reference context where the image appeared at position 0 (no prefix). |

The error norm at each (layer, token) is:

```
error_norm[l][t] = metric_fn(h_reuse[l][t] - h_recompute[l][t])
```

where `h` is the hidden state at the **output** of transformer block `l` for
token `t`, and `metric_fn` is configurable (default: L2 norm, MSE available).

The measurement covers both the **prefill** span (all prompt tokens) and a
configurable number of **greedy decode** steps.

---

## Requirements

| Package | Minimum version | Notes |
|---------|-----------------|-------|
| `torch` | 2.4.0 | CUDA recommended |
| `transformers` | 4.51.0 | Must include `Qwen2_5_VLForConditionalGeneration` |
| `datasets` | any recent | For loading MMMU-Pro |
| `qwen_vl_utils` | — | From the Qwen2.5-VL model card / HuggingFace |
| `matplotlib` | — | For plot generation |
| `numpy` | — | For numerical output |

Install the project dependencies first:

```bash
pip install -e .
pip install datasets qwen-vl-utils matplotlib
```

---

## Model path

The experiment defaults to `~/huggingface/Qwen2.5-VL-3B-Instruct`.
Download the model weights with:

```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir ~/huggingface/Qwen2.5-VL-3B-Instruct
```

Override the path with `--model_path`.

---

## Dataset path

The MMMU-Pro Vision subset is loaded from HuggingFace Hub by default:

```
MMMU/MMMU_Pro  (config: "vision",  split: "test")
```

To use a local copy, pass `--dataset_path /path/to/mmmu_pro`.
The directory must be a HuggingFace `datasets`-compatible format (e.g., created
with `datasets.save_to_disk`).

---

## Running the experiment

```bash
# Minimal (uses defaults: L2, 20 decode steps, sample 0)
python experiments/error_norm_analysis.py \
    --model_path ~/huggingface/Qwen2.5-VL-3B-Instruct

# Full options
python experiments/error_norm_analysis.py \
    --model_path     ~/huggingface/Qwen2.5-VL-3B-Instruct \
    --dataset_path   /data/mmmu_pro \
    --dataset_split  test \
    --sample_index   0 \
    --output_dir     outputs/error_norm \
    --metric         l2 \
    --decode_steps   20 \
    --device         cuda
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `~/huggingface/Qwen2.5-VL-3B-Instruct` | Path to model directory |
| `--dataset_path` | *(HF Hub)* | Local MMMU-Pro dataset directory |
| `--dataset_split` | `test` | Dataset split |
| `--sample_index` | `0` | Zero-based sample index |
| `--output_dir` | `outputs/error_norm` | Where to save outputs |
| `--metric` | `l2` | Error metric: `l2` or `mse` |
| `--decode_steps` | `20` | Number of greedy decode steps |
| `--device` | `cuda` / `cpu` | PyTorch device |

---

## Output locations

After the run, `--output_dir` (default `outputs/error_norm/`) contains:

```
outputs/error_norm/
├── error_matrix.png        # Heatmap: all layers × all tokens
├── error_norms.npy         # Raw float32 array [num_layers, total_tokens]
├── metadata.json           # Sequence lengths, span indices, metric name
└── per_layer/
    ├── layer_00.png        # Per-layer: error norm vs token index
    ├── layer_01.png
    └── …
```

### Vertical dashed lines in plots

Each plot marks three region boundaries:

| Colour | Meaning |
|--------|---------|
| Green dashed | Start of image token span |
| Red dashed | End of image token span |
| Orange dashed | Prefill → decode boundary |

### Raw data format

`error_norms.npy` is a float32 NumPy array with shape
`[num_layers, prefill_len + decode_steps]`.  Columns `0..prefill_len-1`
correspond to prefill positions; columns `prefill_len..` correspond to decode
steps.

`metadata.json` stores:

```json
{
  "prefill_len":  <int>,
  "decode_len":   <int>,
  "image_start":  <int>,
  "image_end":    <int>,
  "num_layers":   <int>,
  "metric":       "l2"
}
```

---

## Extending the experiment

### Swapping the error metric

Pass `--metric mse` to switch to per-token mean-squared error.

To add a custom metric, register it in `METRICS` at the top of the script:

```python
# experiments/error_norm_analysis.py
METRICS["cosine"] = lambda a, b: 1 - torch.nn.functional.cosine_similarity(
    a.float(), b.float(), dim=-1
)
```

### Adding recomputation-ratio curves (future work)

The plotting functions (`plot_per_layer`, `plot_matrix`) are designed to be
called independently of the recomputation source.  To overlay multiple
recompute-ratio curves on the same plot, call `plot_per_layer` once per ratio
with different `error_norms` arrays and merge the Axes objects.

A placeholder configuration dict can be used to store recompute-ratio settings
for future integration:

```python
RECOMPUTE_CONFIGS = {
    "ratio_0.0":  {"recompute_layers": []},
    "ratio_0.05": {"recompute_layers": [0, 1, 2]},
    "ratio_0.10": {"recompute_layers": list(range(6))},
}
```

### Using a different sample

```bash
python experiments/error_norm_analysis.py --sample_index 5
```

---

## Expected output

On a GPU with `Qwen2.5-VL-3B-Instruct` and a typical MMMU-Pro Vision sample:

- The image span contains several hundred image tokens.
- Error norms are **near-zero for text tokens before the image** (their
  position IDs are unchanged between the two passes).
- Error norms are **non-zero starting at the image span** (positional mismatch
  for image tokens propagates to all downstream tokens).
- Error generally **accumulates with layer depth** (consistent with Figure 2 of
  the VLCache paper).
- Error during **decode steps** reflects the propagated error from the wrong
  image K/V in the cache.
