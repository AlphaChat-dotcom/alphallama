#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two AI_TF_DUMP files (engine.cpp [AI_TF]) by decode position, bitwise.

    tf_compare.py plain.tf mtp.tf

Positions are compared only while both runs FED the same token there (with AI_TF_FORCE the MTP run's drafts are
the plain run's tokens, so that holds up to the first row whose argmax differs). For each compared position it
reports whether the logits row is bit-identical, max |dlogit| and whether the argmax agrees. The first position
whose row is not bit-identical is where verify stops being plain decode.
"""
import struct
import sys

import numpy as np


def load(path):
    out = {}
    with open(path, 'rb') as f:
        while True:
            h = f.read(24)
            if len(h) < 24:
                break
            pos, tok, am, V = struct.unpack('<qiiq', h)
            row = np.frombuffer(f.read(4 * V), dtype='<f4')
            out[pos] = (tok, am, row)
    return out


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    ident = 0
    first_diff = None
    first_flip = None
    worst = 0.0
    for p in common:
        ta, ma, ra = a[p]
        tb, mb, rb = b[p]
        if ta != tb:
            print('position %d: fed tokens differ (%d vs %d) -- comparison stops here' % (p, ta, tb))
            break
        same = np.array_equal(ra.view(np.uint32), rb.view(np.uint32))
        d = float(np.abs(ra - rb).max())
        worst = max(worst, d)
        if same:
            ident += 1
        elif first_diff is None:
            first_diff = (p, d)
        if ma != mb and first_flip is None:
            first_flip = (p, ma, mb)
    n = len(common)
    print('positions in both: %d | bit-identical rows: %d | max |dlogit| %.3e' % (n, ident, worst))
    print('first non-identical row: %s' % ('none' if first_diff is None else 'pos %d (max |d| %.3e)' % first_diff))
    print('first argmax flip: %s' % ('none' if first_flip is None else 'pos %d (%d vs %d)' % first_flip))


if __name__ == '__main__':
    main()
