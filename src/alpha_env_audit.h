#pragma once

// ALPHA_ENV_AUDIT: warn about ALPHA_* environment variables that are set in the environment but that
// no loaded code can read. Such a variable is a silent no-op (e.g. code compiled into an object that
// is never linked), and the operator would otherwise conclude the feature was exercised.
//
// Method: a getenv("ALPHA_X") call requires the literal "ALPHA_X\0" in some loaded object's read-only
// data. If that string is absent from every loaded image, nothing can read the variable. This tests the
// artifact rather than the source, so it catches compiled-but-unlinked code as well as typos.
//
// Call once, as late in startup as practical: after backends/shared objects are loaded, since a lazily
// dlopen'd backend (e.g. libggml-cuda.so) carries its own flag strings.
//
// Scope: only variables actually set in the environment are reported, so the output is bounded by
// what the operator typed, not by the size of the flag universe.
//
// Caveat: with GGML_BACKEND_DL=ON (default OFF) backends arrive via dlopen; call this after backend
// registration or a not-yet-loaded backend's flag reads as unknown. Both call sites are after model
// load, which forces registration of the device the model uses.
//
// Set ALPHA_NO_ENV_AUDIT=1 to silence. Linux-only; a no-op elsewhere.
void alpha_env_audit(void);
