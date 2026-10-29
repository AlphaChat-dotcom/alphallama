# Contributing

Thanks for your interest in AlphaLlama. Bug reports, fixes and improvements are welcome.

## Reporting bugs

Open an issue with:

- the exact command line and environment variables you used,
- the model file (name, quantization, source),
- GPU, driver and CUDA version, OS, and how much system RAM you have,
- the server or console log around the problem.

For performance reports, say what you measured (decode-only tok/s or end-to-end time), the prompt and generation
length, and whether the run was warm (same prompt repeated) or cold.

## Pull requests

- Keep changes focused; one topic per pull request.
- Follow the style of the surrounding code (see [.clang-format](.clang-format)); comments should explain what the
  code does and why, not its history.
- MoE and offload changes must stay generic: use model metadata, not checks for a specific model.
- New optional behaviour goes behind a default-off setting until it is shown to help.
- If you change decoding, expert caching or kernels, check that greedy output is unchanged with default settings
  and include before/after speed numbers measured on the same machine.
- Run `ctest --test-dir build` (and `test-backend-ops` for ggml changes) before submitting.

Changes that are not specific to this fork are usually better sent to [llama.cpp](https://github.com/ggml-org/llama.cpp)
first; we merge from upstream regularly.

## Developer Certificate of Origin

Contributions are accepted under the license of the files they change: the MIT License, or the Apache License 2.0
for the parts listed in [NOTICE](NOTICE) (their files carry an SPDX header). Each commit must be signed off to certify the
[Developer Certificate of Origin](https://developercertificate.org/):

```sh
git commit -s -m "your message"
```

which adds a `Signed-off-by: Your Name <you@example.com>` line with your real name.
