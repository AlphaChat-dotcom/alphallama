# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's private vulnerability reporting
(the "Report a vulnerability" button on the repository's Security tab). Do not open a public issue.

Include a description, the affected version or commit, and steps or files that reproduce the problem.
We aim to acknowledge reports within a week and ask for up to 90 days to release a fix before public disclosure.

## Scope

In scope: the inference library (`src/`, `ggml/`, `gguf-py/`), the servers (`tools/server/`, `tools/serve/`) and
the command-line tools in this repository.

Issues in llama.cpp code that also affect upstream should be reported to
[llama.cpp](https://github.com/ggml-org/llama.cpp/security) as well.

## Using AlphaLlama safely

- **Models are code-adjacent input.** Only load GGUF files from sources you trust; malformed files can trigger
  bugs in the loader.
- **The servers are not hardened for the public internet.** Bind to localhost or put them behind an
  authenticating reverse proxy. The `alphallama` server can require a key with `--api-key`;
  `alphallama-serve` has no built-in authentication.
- **Tool calling and retrieval** can expose local files or data placed in a corpus directory to anyone who can
  reach the server. Index only what all users of the server may read.
- **Multi-user deployments** share one process; requests are not isolated from each other at the memory level.
