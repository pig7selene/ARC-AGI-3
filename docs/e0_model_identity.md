# E0 model identity

Status: **checkpoint identity resolved; runtime metadata still captured at E1
startup**.  The current E0 high-score route was confirmed to use the Kaggle
model below.  The older public Duck alias is retained only as historical
forensics and is not an E1 fallback.

| Field | Value | Evidence / confidence |
|---|---|---|
| DISPLAY NAME | `Duck v12 + Qwen3.8-27B-FP8 Repacked` | User-provided E0 submission name; display metadata only. |
| ACTUAL MODEL PATH | `/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` | User-confirmed exact Kaggle HF-compatible checkpoint path. |
| MODEL CONFIG IDENTITY | `Qwen/Qwen3.8-27B-FP8` | User-confirmed model-card identity. Local `config.json` fields are captured at E1 startup. |
| SERVED MODEL NAME | E1 defaults to the exact filesystem path; E0 served name **UNKNOWN** | E1 is explicit and reproducible; the E0 benchmark has no `/v1/models` response. |
| Model revision / commit / checkpoint hash | **UNKNOWN** | Not included in the user-confirmed model-card/path evidence; E1 metadata records it as unknown unless a manifest is present. |
| Model architecture | **UNKNOWN until E1 config inspection** | E1 startup reads `config.json` `architectures` and `model_type`. |
| Tokenizer path / revision | **UNKNOWN until E1 inspection** | E1 startup records the local tokenizer directory when marker files are present. |
| Quantization format | `FP8` in confirmed model-card/name; file-level metadata **UNKNOWN until E1 inspection** | E1 startup records `quantization_config` and `torch_dtype`. |
| Context / serving config | 32,768 max context in the public Duck config | Public config evidence; see [`e0_inference_config.md`](e0_inference_config.md). |

## Evidence boundary

`/Users/infiniteejl/Downloads/benchmark.json` records the solver label,
actions, usage totals, and timing, but no served-model response or vLLM log.
The user-confirmed Kaggle input now locks the E1 target; the repository's E1
launcher defaults to that path and permits a different model only through an
explicit `E1_MODEL_ID` override.

The public source/config is useful for reconstructing the intended serving
contract.  `vrfai/Qwen3.6-27B-FP8` is a historical Duck config identity and is
not used as an E1 fallback.  The current E1 target is the confirmed
`Qwen/Qwen3.8-27B-FP8` checkpoint above.

## Runtime metadata captured by E1

At E1 startup, the launcher/runner prints and writes the following without
guessing missing values:

1. Mounted input path and served model name.
2. `config.json` architectures/model type and quantization fields.
3. Tokenizer path, vLLM version, max model length, and sampling.
4. Thinking/preserve-thinking and reasoning/tool parser settings.
5. Local `/v1/models` response, when available.

Any field not present in the checkpoint or server response is written as
`UNKNOWN`; it is not inferred from a model name.

## Sources

- [Duck public inference config](https://github.com/Tufalabs/duck-harness/blob/main/ARC3-Inference/configs/inference.json)
- [Duck public Kaggle notebook](https://github.com/Tufalabs/duck-harness/blob/main/taaf-duck-harness-kaggle-share.ipynb)
- Local E0 artifact: `benchmark.json` supplied outside this repository
