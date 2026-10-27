#pragma once

// Alpha-MoLE: Runtime Registry
//
// Extends compiled-in architecture support with a JSON registry file
// at ~/.alphachat/registry.json. Allows adding new tensor aliases,
// LoRA patterns, and architecture fallbacks without rebuilding.
//
// The registry is fetched from the platform API via `alphachat update`
// and merged with compiled-in defaults at runtime.

#include <string>
#include <vector>
#include <map>

struct alpha_registry_lora_target {
    std::vector<std::string> patterns;
    bool fused = false;
    int fused_parts = 0;
};

struct alpha_registry_arch {
    std::string name;
    std::string fallback_arch;  // if set, use this arch's loader (e.g., "llama")
    std::vector<std::pair<std::string, std::string>> tensor_aliases;  // primary -> alias
    struct {
        alpha_registry_lora_target q, k, v, o, gate, up, down;
    } lora;
    bool has_lora = false;
};

struct alpha_registry {
    int version = 0;
    std::string updated_at;
    std::map<std::string, alpha_registry_arch> architectures;
};

// Load registry from ~/.alphachat/registry.json (returns empty if not found)
alpha_registry alpha_registry_load();

// Save registry to ~/.alphachat/registry.json
bool alpha_registry_save(const alpha_registry & reg, const std::string & path = "");

// Check if an architecture is in the runtime registry
bool alpha_registry_has_arch(const alpha_registry & reg, const std::string & arch);

// Get tensor aliases for an architecture (empty if not found)
std::vector<std::pair<std::string, std::string>> alpha_registry_get_aliases(
    const alpha_registry & reg, const std::string & arch);

// Get fallback architecture name (empty if none)
std::string alpha_registry_get_fallback(const alpha_registry & reg, const std::string & arch);

// Get default registry path
std::string alpha_registry_default_path();
