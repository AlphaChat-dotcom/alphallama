#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q4x_compare.py: correctness check -- teacher-forced logits of ai-q4x-run vs the alphallama oracle (ai-ref --score).

    python3 q4x_compare.py ref.logits ours.logits [--tokens p.i32]

Both files: i64 n, i64 V, then n x V float32 (ai-ref's --score format). Reports per-position top-1 agreement, mean and
max KL(ref || ours), max |dlogit|, and -- with --tokens -- the NLL of the actual next token under both (the PPL delta).
Pass: top-1 >= 99.5%, KL small, PPL within noise of the oracle. Not bitwise: different kernels.
"""
import argparse
import numpy as np


def load(path):
    with open(path, 'rb') as f:
        n, V = np.frombuffer(f.read(16), dtype=np.int64)
        return np.memmap(path, dtype=np.float32, mode='r', offset=16, shape=(int(n), int(V)))


def logsoftmax(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


ap = argparse.ArgumentParser()
ap.add_argument('ref')
ap.add_argument('ours')
ap.add_argument('--tokens')
a = ap.parse_args()
R, O = load(a.ref), load(a.ours)
assert R.shape == O.shape, (R.shape, O.shape)
toks = np.fromfile(a.tokens, dtype=np.int32) if a.tokens else None
n = R.shape[0]
agree, kls, maxd, nll_r, nll_o = 0, [], 0.0, [], []
for i in range(n):
    r, o = np.asarray(R[i], dtype=np.float64), np.asarray(O[i], dtype=np.float64)
    if not (np.isfinite(r).all() and np.isfinite(o).all()):
        print('non-finite logits at position', i)
        continue
    agree += int(r.argmax() == o.argmax())
    lr, lo = logsoftmax(r), logsoftmax(o)
    kls.append(float((np.exp(lr) * (lr - lo)).sum()))
    maxd = max(maxd, float(np.abs(r - o).max()))
    if toks is not None and i + 1 < len(toks):
        nll_r.append(-lr[toks[i + 1]])
        nll_o.append(-lo[toks[i + 1]])
print('positions %d | top-1 agreement %.2f%% | KL mean %.3e max %.3e | max |dlogit| %.4f'
      % (n, 100.0 * agree / n, np.mean(kls), np.max(kls), maxd))
if nll_r:
    d = np.array(nll_o) - np.array(nll_r)
    print('next-token NLL: oracle %.4f ours %.4f (delta %+.4f, paired SE %.4f) -> PPL %.3f vs %.3f'
          % (np.mean(nll_r), np.mean(nll_o), d.mean(), d.std(ddof=1) / np.sqrt(len(d)),
             np.exp(np.mean(nll_r)), np.exp(np.mean(nll_o))))
