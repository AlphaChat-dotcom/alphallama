#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate: AlphaInfer vs the llama.cpp forward on the same tokens.

  compare.py logits REF.logits AI.logits        teacher-forced top-1 agreement, KL, max |dlogit|
  compare.py gen REF_gen.i32 AI_gen.i32          free-running greedy: matched prefix length

Logit files: int64 n, int64 V, then n*V float32 rows.
"""
import sys

import numpy as np


def read_logits(path):
    with open(path, "rb") as f:
        n, v = np.frombuffer(f.read(16), dtype=np.int64)
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(int(n), int(v))


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def cmd_logits(ref_path, ai_path):
    ref, ai = read_logits(ref_path), read_logits(ai_path)
    if ref.shape != ai.shape:
        sys.exit(f"shape mismatch {ref.shape} vs {ai.shape}")
    n = ref.shape[0]
    top_ref, top_ai = ref.argmax(1), ai.argmax(1)
    agree = (top_ref == top_ai)
    kls, maxd = [], []
    for i in range(0, n, 64):
        lr, la = log_softmax(ref[i:i + 64]), log_softmax(ai[i:i + 64])
        kls.append((np.exp(lr) * (lr - la)).sum(1))
        maxd.append(np.abs(ref[i:i + 64] - ai[i:i + 64]).max(1))
    kl, md = np.concatenate(kls), np.concatenate(maxd)
    # a disagreement is a tie flip when the ref's own top-2 gap is tiny
    srt = np.sort(ref, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    flips = np.where(~agree)[0]
    print(f"positions            {n}")
    print(f"top-1 agreement      {agree.mean() * 100:.3f}%  ({agree.sum()}/{n})")
    print(f"mean KL(ref||ai)     {kl.mean():.3e}   p99 {np.percentile(kl, 99):.3e}   max {kl.max():.3e}")
    print(f"max |dlogit|         mean {md.mean():.4f}   max {md.max():.4f}")
    if len(flips):
        print(f"disagreements at     {flips[:20].tolist()}  ref top-2 gap there: "
              f"{np.round(gap[flips[:20]], 4).tolist()}")


def cmd_gen(ref_path, ai_path):
    r = np.fromfile(ref_path, dtype=np.int32)
    a = np.fromfile(ai_path, dtype=np.int32)
    n = min(len(r), len(a))
    match = n
    for i in range(n):
        if r[i] != a[i]:
            match = i
            break
    print(f"ref {len(r)} tokens, ai {len(a)} tokens, matched prefix {match}/{n}")
    if match < n:
        print(f"first divergence at {match}: ref {r[match]} vs ai {a[match]}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("logits", "gen"):
        sys.exit(__doc__)
    (cmd_logits if sys.argv[1] == "logits" else cmd_gen)(sys.argv[2], sys.argv[3])
