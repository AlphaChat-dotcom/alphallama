#pragma once

// Oracle model embedding support.
// When ALPHA_ORACLE_EMBEDDED is defined, the oracle-0.8b.gguf is linked
// into the binary and can be loaded without an external file.

#include <cstddef>
#include <cstdint>

namespace oracle_embed {

// Returns true if the oracle model is embedded in this binary.
bool available();

// Returns pointer to embedded oracle data (nullptr if not embedded).
const uint8_t * data();

// Returns size of embedded oracle data (0 if not embedded).
size_t size();

} // namespace oracle_embed
