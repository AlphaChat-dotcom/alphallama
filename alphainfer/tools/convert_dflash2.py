#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DFlash2 drafter (z-lab, HF safetensors BF16) -> GGUF, arch "dflash2".
#
#   convert_dflash2.py <hf dir> <out.gguf> [--mat q8_0|bf16]
#
# --mat q8_0 (default): the 2-D projection matrices become q8_0 (8-bit weights, groups of 32, fp16 scale).
# --mat bf16: every tensor stays BF16 (lossless), for validating the forward pass against the z-lab reference.
# Norms and conv base kernels are F32; the selector codebooks stay BF16 (lossless) in both modes.
# Every tensor is read back and compared with its source: BF16/F32 must be bit-exact, q8_0 must be within NMSE 1e-4.
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('hf'); ap.add_argument('out')
ap.add_argument('--mat', choices=['q8_0', 'bf16'], default='q8_0')
ap.add_argument('--gguf-py', default=os.environ.get('GGUF_PY', ''))
a = ap.parse_args()
if a.gguf_py:
    sys.path.insert(0, a.gguf_py)
import gguf
from gguf.quants import quantize, dequantize
from safetensors import safe_open
import torch

cfg = json.load(open(os.path.join(a.hf, 'config.json')))
dc = cfg['dflash_config']
n_layer = cfg['num_hidden_layers']
# DFlash v1 (DFlashDraftModel, e.g. z-lab/Qwen3.6-35B-A3B-DFlash) is the base of DFlash2: no dynamic convs, no
# candidate selector (drafts = argmax of the target head), and per-layer attention -- sliding layers causal with a
# window, full layers bidirectional and unwindowed, when the config has no is_causal (dflash/model.py
# Qwen3DFlashAttention). Same GGUF arch "dflash2"; the variant is in the metadata below.
v2 = 'DFlash2DraftModel' in cfg.get('architectures', []) or 'selector_rank' in dc
ltypes = cfg.get('layer_types') or ['full_attention'] * n_layer
assert all(t in ('sliding_attention', 'full_attention') for t in ltypes), ltypes
sliding = [1 if t == 'sliding_attention' else 0 for t in ltypes]
causal = [int(bool(cfg['is_causal'])) if cfg.get('is_causal') is not None else s_ for s_ in sliding]
assert not cfg['attention_bias'] and cfg['hidden_act'] == 'silu', 'spec assumes no bias, silu'
assert cfg['rope_parameters']['rope_type'] == 'default', cfg['rope_parameters']

T = gguf.GGMLQuantizationType
MAT = T.Q8_0 if a.mat == 'q8_0' else T.BF16

# HF name -> (GGUF name, storage type). Layer tensors are expanded per layer below.
LAYER = {
    'input_layernorm.weight':               ('attn_norm.weight',      T.F32),
    'post_attention_layernorm.weight':      ('ffn_norm.weight',       T.F32),
    'self_attn.q_proj.weight':              ('attn_q.weight',         MAT),
    'self_attn.k_proj.weight':              ('attn_k.weight',         MAT),
    'self_attn.v_proj.weight':              ('attn_v.weight',         MAT),
    'self_attn.o_proj.weight':              ('attn_output.weight',    MAT),
    'self_attn.q_norm.weight':              ('attn_q_norm.weight',    T.F32),
    'self_attn.k_norm.weight':              ('attn_k_norm.weight',    T.F32),
    'mlp.gate_proj.weight':                 ('ffn_gate.weight',       MAT),
    'mlp.up_proj.weight':                   ('ffn_up.weight',         MAT),
    'mlp.down_proj.weight':                 ('ffn_down.weight',       MAT),
}
if v2:
    LAYER.update({
        'attention_conv.base_kernel':           ('attn_conv.base',        T.F32),
        'attention_conv.kernel_projection.weight': ('attn_conv.proj.weight', MAT),
        'mlp_conv.base_kernel':                 ('ffn_conv.base',         T.F32),
        'mlp_conv.kernel_projection.weight':    ('ffn_conv.proj.weight',  MAT),
    })
GLOBAL = {
    'fc.weight':                                   ('ctx_fc.weight',      MAT),
    'hidden_norm.weight':                          ('ctx_norm.weight',    T.F32),
    'norm.weight':                                 ('output_norm.weight', T.F32),
}
if v2:
    GLOBAL.update({
        'candidate_selector.hidden_projection.weight': ('sel_proj.weight',    MAT),
        'candidate_selector.predecessor_codebook':     ('sel_pred_cb',        T.BF16),
        'candidate_selector.successor_codebook':       ('sel_succ_cb',        T.BF16),
    })
MAP = dict(GLOBAL)
for il in range(n_layer):
    for k, (g, t) in LAYER.items():
        MAP['layers.%d.%s' % (il, k)] = ('blk.%d.%s' % (il, g), t)

st = safe_open(os.path.join(a.hf, 'model.safetensors'), framework='pt')
keys = list(st.keys())
# accept an optional "model." prefix but require an exact one-to-one cover of the map
strip = lambda k: k[len('model.'):] if k.startswith('model.') else k
src = {strip(k): k for k in keys}
missing = sorted(set(MAP) - set(src)); extra = sorted(set(src) - set(MAP))
if missing or extra:
    sys.exit('tensor set mismatch: missing %s extra %s' % (missing[:8], extra[:8]))

w = gguf.GGUFWriter(a.out, 'dflash2')
w.add_name(cfg.get('_name_or_path') or ('DFlash2' if v2 else 'DFlash'))
A = 'dflash2.'
w.add_uint32(A + 'block_count', n_layer)
w.add_uint32(A + 'embedding_length', cfg['hidden_size'])
w.add_uint32(A + 'feed_forward_length', cfg['intermediate_size'])
w.add_uint32(A + 'attention.head_count', cfg['num_attention_heads'])
w.add_uint32(A + 'attention.head_count_kv', cfg['num_key_value_heads'])
w.add_uint32(A + 'attention.key_length', cfg['head_dim'])
w.add_uint32(A + 'attention.value_length', cfg['head_dim'])
w.add_float32(A + 'attention.layer_norm_rms_epsilon', cfg['rms_norm_eps'])
w.add_uint32(A + 'attention.sliding_window', cfg['sliding_window'])
w.add_float32(A + 'rope.freq_base', float(cfg['rope_parameters']['rope_theta']))
w.add_uint32(A + 'vocab_size', cfg['vocab_size'])
w.add_uint32(A + 'block_size', dc['block_size'])
w.add_uint32(A + 'mask_token_id', dc['mask_token_id'])
w.add_uint32(A + 'variant', 2 if v2 else 1)
if v2:
    w.add_uint32(A + 'conv_group_size', dc['conv_group_size'])
    w.add_uint32(A + 'conv_kernel_size', dc['conv_kernel_size'])
    w.add_uint32(A + 'selector_rank', dc['selector_rank'])
    w.add_uint32(A + 'selector_top_k', dc['selector_top_k'])
w.add_array(A + 'layer_sliding', sliding)
w.add_array(A + 'layer_causal', causal)
w.add_uint32(A + 'num_target_layers', cfg['num_target_layers'])
w.add_array(A + 'target_layer_ids', [int(x) for x in dc['target_layer_ids']])

def f32_of(hf):
    x = st.get_tensor(src[hf])
    assert x.dtype == torch.bfloat16, (hf, x.dtype)   # the checkpoint is all BF16
    return x.float().numpy()

stored = {}
total = 0
for hf, (g, t) in MAP.items():
    x = f32_of(hf)
    if t == T.Q8_0 and x.shape[-1] % 32:
        sys.exit('%s: row %d not a multiple of 32' % (hf, x.shape[-1]))
    q = x if t == T.F32 else quantize(x, t)
    # quantize() returns a byte-shaped uint8 array; gguf-py derives the element shape from it
    w.add_tensor(g, q, raw_dtype=None if t == T.F32 else t)
    stored[g] = (hf, t, x.shape)
    total += q.nbytes
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print('wrote %s: %d tensors, %.2f GB tensor data, matrices %s' % (a.out, len(stored), total / 1e9, MAT.name))

# read back every tensor and compare with its source
r = gguf.GGUFReader(a.out)
assert str(bytes(r.fields['general.architecture'].parts[-1]), 'utf-8') == 'dflash2'
seen = 0
worst = 0.0
for t in r.tensors:
    hf, ty, shape = stored[t.name]
    assert t.tensor_type == ty, (t.name, t.tensor_type, ty)
    ne = [int(v) for v in t.shape]
    assert ne == list(reversed(shape)), (t.name, ne, shape)
    back = dequantize(np.asarray(t.data), t.tensor_type).reshape(shape)
    ref = f32_of(hf)
    if ty in (T.F32, T.BF16):
        if not np.array_equal(back, ref):
            sys.exit('%s: %s round trip NOT bit-exact' % (t.name, ty.name))
    else:
        nmse = float(((back - ref) ** 2).mean() / max((ref ** 2).mean(), 1e-30))
        worst = max(worst, nmse)
        if nmse > 1e-4:
            sys.exit('%s: q8_0 NMSE %.3g > 1e-4' % (t.name, nmse))
    seen += 1
assert seen == len(stored), (seen, len(stored))
print('read-back OK: %d/%d tensors; F32/BF16 bit-exact; worst q8_0 NMSE %.3g' % (seen, len(stored), worst))
