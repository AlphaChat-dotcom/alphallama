#include "alpha_registry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

std::string alpha_registry_default_path() {
    const char * home = getenv("HOME");
    if (!home) home = getenv("USERPROFILE");
    if (!home) return "";
    return std::string(home) + "/.alphachat/registry.json";
}

// Minimal JSON string extractor (no dependency on nlohmann)
// Handles the simple registry format without a full parser
static std::string json_get_string(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find("\"", pos + search.size() + 1);  // skip colon, find value start
    if (pos == std::string::npos) return "";
    auto end = json.find("\"", pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

static int json_get_int(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return 0;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return 0;
    return atoi(json.c_str() + pos + 1);
}

// Extract a JSON array of string pairs: [["a","b"],["c","d"]]
static std::vector<std::pair<std::string, std::string>> json_get_alias_array(
        const std::string & json, const std::string & key) {
    std::vector<std::pair<std::string, std::string>> result;
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return result;

    // Find the opening [
    pos = json.find("[", pos);
    if (pos == std::string::npos) return result;

    // Find matching ]
    auto end = json.find("]", pos);
    if (end == std::string::npos) return result;

    std::string arr = json.substr(pos, end - pos + 1);

    // Parse inner arrays: ["a","b"]
    size_t i = 0;
    while (i < arr.size()) {
        auto inner_start = arr.find("[", i + 1);
        if (inner_start == std::string::npos) break;
        auto inner_end = arr.find("]", inner_start);
        if (inner_end == std::string::npos) break;

        std::string inner = arr.substr(inner_start + 1, inner_end - inner_start - 1);
        // Extract two quoted strings
        auto q1 = inner.find("\"");
        auto q2 = inner.find("\"", q1 + 1);
        auto q3 = inner.find("\"", q2 + 1);
        auto q4 = inner.find("\"", q3 + 1);
        if (q1 != std::string::npos && q4 != std::string::npos) {
            result.emplace_back(inner.substr(q1+1, q2-q1-1), inner.substr(q3+1, q4-q3-1));
        }
        i = inner_end + 1;
    }
    return result;
}

// Extract patterns array from LoRA target: {"patterns":["a","b"]}
static std::vector<std::string> json_get_string_array(const std::string & json, const std::string & key) {
    std::vector<std::string> result;
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return result;
    pos = json.find("[", pos);
    if (pos == std::string::npos) return result;
    auto end = json.find("]", pos);
    if (end == std::string::npos) return result;

    std::string arr = json.substr(pos + 1, end - pos - 1);
    size_t i = 0;
    while (i < arr.size()) {
        auto q1 = arr.find("\"", i);
        if (q1 == std::string::npos) break;
        auto q2 = arr.find("\"", q1 + 1);
        if (q2 == std::string::npos) break;
        result.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return result;
}

// Find a JSON object block by key name
static std::string json_get_object(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find("{", pos);
    if (pos == std::string::npos) return "";

    int depth = 0;
    size_t end = pos;
    for (; end < json.size(); end++) {
        if (json[end] == '{') depth++;
        else if (json[end] == '}') { depth--; if (depth == 0) break; }
    }
    return json.substr(pos, end - pos + 1);
}

alpha_registry alpha_registry_load() {
    alpha_registry reg;
    std::string path = alpha_registry_default_path();
    if (path.empty() || !fs::exists(path)) return reg;

    try {
        std::ifstream f(path);
        if (!f.is_open()) return reg;
        std::stringstream ss;
        ss << f.rdbuf();
        std::string content = ss.str();

        reg.version = json_get_int(content, "version");
        reg.updated_at = json_get_string(content, "updated_at");

        std::string archs = json_get_object(content, "architectures");
        if (archs.empty()) return reg;

        // Find each architecture entry
        size_t pos = 0;
        while (pos < archs.size()) {
            // Find next quoted key
            auto q1 = archs.find("\"", pos + 1);
            if (q1 == std::string::npos) break;
            auto q2 = archs.find("\"", q1 + 1);
            if (q2 == std::string::npos) break;

            std::string arch_name = archs.substr(q1 + 1, q2 - q1 - 1);
            // Skip non-architecture keys
            if (arch_name == "version" || arch_name == "updated_at" || arch_name == "architectures") {
                pos = q2 + 1;
                continue;
            }

            std::string arch_obj = json_get_object(archs.substr(q2), "");
            if (arch_obj.empty()) { pos = q2 + 1; continue; }

            // Actually get the object starting from the colon after the key
            auto colon = archs.find(":", q2);
            auto brace = archs.find("{", colon);
            if (brace == std::string::npos) { pos = q2 + 1; continue; }

            int depth = 0;
            size_t end = brace;
            for (; end < archs.size(); end++) {
                if (archs[end] == '{') depth++;
                else if (archs[end] == '}') { depth--; if (depth == 0) break; }
            }
            arch_obj = archs.substr(brace, end - brace + 1);

            alpha_registry_arch arch;
            arch.name = arch_name;
            arch.fallback_arch = json_get_string(arch_obj, "fallback_arch");
            arch.tensor_aliases = json_get_alias_array(arch_obj, "tensor_aliases");

            // Parse LoRA targets
            std::string lora_obj = json_get_object(arch_obj, "lora");
            if (!lora_obj.empty()) {
                arch.has_lora = true;
                auto parse_target = [&](const std::string & name) -> alpha_registry_lora_target {
                    alpha_registry_lora_target t;
                    std::string target_obj = json_get_object(lora_obj, name);
                    if (!target_obj.empty()) {
                        t.patterns = json_get_string_array(target_obj, "patterns");
                    }
                    return t;
                };
                arch.lora.q    = parse_target("q");
                arch.lora.k    = parse_target("k");
                arch.lora.v    = parse_target("v");
                arch.lora.o    = parse_target("o");
                arch.lora.gate = parse_target("gate");
                arch.lora.up   = parse_target("up");
                arch.lora.down = parse_target("down");
            }

            reg.architectures[arch_name] = std::move(arch);
            pos = end + 1;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "alpha_registry: failed to parse %s: %s\n", path.c_str(), e.what());
    }

    return reg;
}

bool alpha_registry_save(const alpha_registry & reg, const std::string & path_override) {
    std::string path = path_override.empty() ? alpha_registry_default_path() : path_override;
    if (path.empty()) return false;

    fs::path dir = fs::path(path).parent_path();
    if (!dir.empty()) fs::create_directories(dir);

    try {
        std::ofstream out(path);
        out << "{\n";
        out << "  \"version\": " << reg.version << ",\n";
        out << "  \"updated_at\": \"" << reg.updated_at << "\",\n";
        out << "  \"architectures\": {\n";

        bool first_arch = true;
        for (auto & [name, arch] : reg.architectures) {
            if (!first_arch) out << ",\n";
            first_arch = false;
            out << "    \"" << name << "\": {\n";
            if (!arch.fallback_arch.empty())
                out << "      \"fallback_arch\": \"" << arch.fallback_arch << "\",\n";
            if (!arch.tensor_aliases.empty()) {
                out << "      \"tensor_aliases\": [";
                for (size_t i = 0; i < arch.tensor_aliases.size(); i++) {
                    if (i > 0) out << ", ";
                    out << "[\"" << arch.tensor_aliases[i].first << "\", \""
                        << arch.tensor_aliases[i].second << "\"]";
                }
                out << "]\n";
            }
            out << "    }";
        }
        out << "\n  }\n}\n";
        return true;
    } catch (...) {
        return false;
    }
}

bool alpha_registry_has_arch(const alpha_registry & reg, const std::string & arch) {
    return reg.architectures.find(arch) != reg.architectures.end();
}

std::vector<std::pair<std::string, std::string>> alpha_registry_get_aliases(
    const alpha_registry & reg, const std::string & arch) {
    auto it = reg.architectures.find(arch);
    if (it == reg.architectures.end()) return {};
    return it->second.tensor_aliases;
}

std::string alpha_registry_get_fallback(const alpha_registry & reg, const std::string & arch) {
    auto it = reg.architectures.find(arch);
    if (it == reg.architectures.end()) return "";
    return it->second.fallback_arch;
}
