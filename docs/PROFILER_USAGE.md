# Quick start (inside Ascend Docker container)
```sh
# Ensure normal Ascend runtime, torch_npu, vllm_ascend custom operator libs
# and msprof binaries are available in this shell.
python3 profile_li.py --mode detailed -impl ops     # For DeepSeek
python3 profile_li.py --help
python3 profile_li.py
python3 profile_li.py --mode simple -impl npu --validate
python3 profile_li.py --mode detailed -impl ops --aic-metrics Default --validate
python3 profile_li.py --mode detailed -impl ops-quant --batch 1 --q_len 1 --kv_len 8192
```
The most important flags are `--mode [simple/detailed/simulator]` and `--impl [ops/npu/ops-quant]`

After a run, inspect `runs/profile-MMDD-HHMM*/summary.txt` and, for detailed captures, the visualize_data.bin relative paths in results.json. Import the .bin into MindStudio Insight.

Compatibility notes: This is a best-effort implementation reviewed against published APIs, not executed on an Ascend NPU here. Operator registration/schema, NPU event timing and msprof/MSTX feature availability must be smoke-tested in the actual container. The ops-quant binding may not exist in newer vLLM-Ascend releases; it intentionally fails with a diagnostic log instead of switching implementations. Full stdout/stderr from failed calls are stored in the run's logs/ directory.

To smoke-test orchestration (without NPU), run `python3 -m unittest discover -s tests -v`.