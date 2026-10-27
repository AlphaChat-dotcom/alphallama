# Caller-driven decode: acceptance tests

Two tests, and the second exists because the first cannot stand alone.

## `ai_api_decode_test.c` — the C ABI acceptance gate

Pure C (`gcc -std=c11 -Wall -Wextra`), linked against `libaicore.a`, so the "C so the ABI does not
depend on either side's flags" promise is exercised rather than asserted. Covers prefill → decode →
`reset_stream` → prefill → decode, the position assert, and every refusal (`logits_mask`,
`n_tokens > n_streams`, decode-before-prefill, out-of-range slot).

The slot-reuse row compares a reused slot against a **fresh engine** on the same prompt, because
`rc == 0` would be decorative — what `reset_stream` buys is that the *second* request is correct.
Its can't-fail row requires the two prompts to produce *different* tokens, else that equality holds
for free.

⚠️ **It is a RELATIVE test.** A defect affecting both arms equally passes it. That is why the anchor
below exists.

## `anchor_driver_equiv.cpp` — the absolute anchor

Drives the **same tokens** through `generate_batch` (the engine's own loop, byte-identity-gated
against the pre-refactor binary) and through `open_streams`/`prefill_one`/`decode_step` in one
process, and requires identical output. Chaining the two gates gives an absolute result:

```
pre-refactor binary       == post-refactor generate_batch   (Stage 1 gate, 3/3 arms)
post-refactor generate_batch == caller-driven path          (this test)
=> caller-driven path == the original engine
```

⭐ **It carries a strength gate on itself.** The first run used arbitrary token ids, the reference
emitted one token twelve times, and two paths agreeing on a constant sequence is a low-information
agreement that is easy to match by accident. It now uses a real tokenized chat prompt and returns
**INCONCLUSIVE rather than PASS** if the reference has fewer than 5 distinct tokens. Last run: 23
distinct tokens over 24, coherent English, identical across both drivers.

## Building

Link line (from `build/CMakeFiles/alphainfer.dir/link.txt`):

```sh
cd ~/alphainfer && FORK=$PWD/third_party/build-fork/bin
gcc -std=c11 -Wall -Wextra -I src -c tools/api-tests/ai_api_decode_test.c -o /tmp/aid.o
g++ -o /tmp/ai_api_decode_test /tmp/aid.o build/libaicore.a \
    -L/usr/local/cuda-12.4/targets/x86_64-linux/lib \
    $FORK/libggml-base.so $FORK/libggml-cuda.so $FORK/libggml.so $FORK/libllama.so \
    /usr/local/cuda-12.4/lib64/libcudart.so -lcudadevrt -ldl -lpthread -lrt -Wl,-rpath,$FORK
```

Both need ~21 GiB free, so stop production on the 3090 first and restore it after.

## `warm_gate.cpp` — eager profile instantiation

The property under test is **not** "`warm_profiles` returns a number" — it is that **zero** profiles
instantiate during serving. That is read off `tm.rounds_recorded`, not parsed out of logs.

```
arm A (no warm):  serve            -> records profiles lazily, mid-serving
arm B (warmed):   warm, then serve -> must record 0
```

⭐ **Can't-fail row, checked before any pass is reported:** arm A must record **> 0**. If the unwarmed
arm also records nothing, "zero during serving" is true for free and arm B proves nothing.

Also requires warming to be **output-neutral** (arm A tokens == arm B tokens), with the same strength
gate as the anchor: an equality over fewer than 5 distinct tokens is not credited.

Last run at ctx 4096 (5-bucket ladder): arm A recorded 1 mid-serving, arm B warmed 5 and recorded 0,
tokens identical, 23 distinct over 24. stderr carried exactly 6 `[ai-profile] recorded` lines = 1 + 5,
so the counter and the log reconcile.
