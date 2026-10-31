#include "server-tools.h"
#include "server-context.h"  // for corpus_tool_context (full definition)

#include <sheredom/subprocess.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include <climits>

namespace fs = std::filesystem;

//
// internal helpers
//

// Locks the server's corpus mutex (if the context has one) for reads of the entity index, linker and
// extractor, which /api/kv_chunk/index updates from other HTTP threads.
static std::unique_lock<std::mutex> corpus_lock(const corpus_tool_context & ctx) {
    return ctx.corpus_mtx ? std::unique_lock<std::mutex>(*ctx.corpus_mtx) : std::unique_lock<std::mutex>();
}

static std::vector<char *> to_cstr_vec(const std::vector<std::string> & v) {
    std::vector<char *> r;
    r.reserve(v.size() + 1);
    for (const auto & s : v) {
        r.push_back(const_cast<char *>(s.c_str()));
    }
    r.push_back(nullptr);
    return r;
}

struct run_proc_result {
    std::string output;
    int  exit_code = -1;
    bool timed_out = false;
};

static run_proc_result run_process(
        const std::vector<std::string> & args,
        size_t max_output,
        int timeout_secs) {
    run_proc_result res;

    subprocess_s proc;
    auto argv = to_cstr_vec(args);

    int options = subprocess_option_no_window
                | subprocess_option_combined_stdout_stderr
                | subprocess_option_inherit_environment
                | subprocess_option_search_user_path;

    if (subprocess_create(argv.data(), options, &proc) != 0) {
        res.output = "failed to spawn process";
        return res;
    }

    std::atomic<bool> done{false};
    std::atomic<bool> timed_out{false};

    std::thread timeout_thread([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_secs);
        while (!done.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                timed_out.store(true);
                subprocess_terminate(&proc);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    FILE * f = subprocess_stdout(&proc);
    std::string output;
    bool truncated = false;
    if (f) {
        char buf[4096];
        while (fgets(buf, sizeof(buf), f) != nullptr) {
            if (!truncated) {
                size_t len = strlen(buf);
                if (output.size() + len <= max_output) {
                    output.append(buf, len);
                } else {
                    output.append(buf, max_output - output.size());
                    truncated = true;
                }
            }
        }
    }

    done.store(true);
    if (timeout_thread.joinable()) {
        timeout_thread.join();
    }

    subprocess_join(&proc, &res.exit_code);
    subprocess_destroy(&proc);

    res.output    = output;
    res.timed_out = timed_out.load();
    if (truncated) {
        res.output += "\n[output truncated]";
    }
    return res;
}

json server_tool::to_json() {
    return {
        {"display_name", display_name},
        {"tool", name},
        {"type", "builtin"},
        {"permissions", json{
            {"write", permission_write}
        }},
        {"definition", get_definition()},
    };
}

//
// read_file: read a file with optional line range and line-number prefix
//

static constexpr size_t SERVER_TOOL_READ_FILE_MAX_SIZE = 16 * 1024; // 16 KB

struct server_tool_read_file : server_tool {
    server_tool_read_file() {
        name = "read_file";
        display_name = "Read file";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read the contents of a file. Optionally specify a 1-based line range. "
                                "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192 ...\")."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",       {{"type", "string"},  {"description", "Path to the file"}}},
                        {"start_line", {{"type", "integer"}, {"description", "First line to read, 1-based (default: 1)"}}},
                        {"end_line",   {{"type", "integer"}, {"description", "Last line to read, 1-based inclusive (default: end of file)"}}},
                        {"append_loc", {{"type", "boolean"}, {"description", "Prefix each line with its line number"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path  = params.at("path").get<std::string>();
        int  start_line   = json_value(params, "start_line", 1);
        int  end_line     = json_value(params, "end_line",  -1); // -1 = no limit
        bool append_loc   = json_value(params, "append_loc", false);

        std::error_code ec;
        uintmax_t file_size = fs::file_size(path, ec);
        if (ec) {
            return {{"error", "cannot stat file: " + ec.message()}};
        }
        if (file_size > SERVER_TOOL_READ_FILE_MAX_SIZE && end_line == -1) {
            return {{"error", string_format(
                "file too large (%zu bytes, max %zu). Use start_line/end_line to read a portion.",
                (size_t)file_size, SERVER_TOOL_READ_FILE_MAX_SIZE)}};
        }

        std::ifstream f(path);
        if (!f) {
            return {{"error", "failed to open file: " + path}};
        }

        std::string result;
        std::string line;
        int lineno = 0;

        while (std::getline(f, line)) {
            lineno++;
            if (lineno < start_line) continue;
            if (end_line != -1 && lineno > end_line) break;

            std::string out_line;
            if (append_loc) {
                out_line = std::to_string(lineno) + "\u2192 " + line + "\n";
            } else {
                out_line = line + "\n";
            }

            if (result.size() + out_line.size() > SERVER_TOOL_READ_FILE_MAX_SIZE) {
                result += "[output truncated]";
                break;
            }
            result += out_line;
        }

        return {{"plain_text_response", result}};
    }
};

//
// file_glob_search: find files matching a glob pattern under a base directory
//

static constexpr size_t SERVER_TOOL_FILE_SEARCH_MAX_RESULTS = 100;

struct server_tool_file_glob_search : server_tool {
    server_tool_file_glob_search() {
        name = "file_glob_search";
        display_name = "File search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Recursively search for files matching a glob pattern under a directory."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Base directory to search in"}}},
                        {"include", {{"type", "string"}, {"description", "Glob pattern for files to include (e.g. \"**/*.cpp\"). Default: **"}}},
                        {"exclude", {{"type", "string"}, {"description", "Glob pattern for files to exclude"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string base    = params.at("path").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));

        std::ostringstream output_text;
        size_t count = 0;

        std::error_code ec;
        for (const auto & entry : fs::recursive_directory_iterator(base,
                fs::directory_options::skip_permission_denied, ec)) {
            if (!entry.is_regular_file()) continue;

            std::string rel = fs::relative(entry.path(), base, ec).string();
            if (ec) continue;
            std::replace(rel.begin(), rel.end(), '\\', '/');

            if (!glob_match(include, rel)) continue;
            if (!exclude.empty() && glob_match(exclude, rel)) continue;

            output_text << entry.path().string() << "\n";
            if (++count >= SERVER_TOOL_FILE_SEARCH_MAX_RESULTS) {
                break;
            }
        }

        output_text << "\n---\nTotal matches: " << count << "\n";

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// grep_search: search for a regex pattern in files
//

static constexpr size_t SERVER_TOOL_GREP_SEARCH_MAX_RESULTS = 100;

struct server_tool_grep_search : server_tool {
    server_tool_grep_search() {
        name = "grep_search";
        display_name = "Grep search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Search for a regex pattern in files under a path. Returns matching lines."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",                {{"type", "string"},  {"description", "File or directory to search in"}}},
                        {"pattern",             {{"type", "string"},  {"description", "Regular expression pattern to search for"}}},
                        {"include",             {{"type", "string"},  {"description", "Glob pattern to filter files (default: **)"}}},
                        {"exclude",             {{"type", "string"},  {"description", "Glob pattern to exclude files"}}},
                        {"return_line_numbers", {{"type", "boolean"}, {"description", "If true, include line numbers in results"}}},
                    }},
                    {"required", json::array({"path", "pattern"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string pat_str = params.at("pattern").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));
        bool show_lineno    = json_value(params, "return_line_numbers", false);

        std::regex pattern;
        try {
            pattern = std::regex(pat_str);
        } catch (const std::regex_error & e) {
            return {{"error", std::string("invalid regex: ") + e.what()}};
        }

        std::ostringstream output_text;
        size_t total = 0;

        auto search_file = [&](const fs::path & fpath) {
            std::ifstream f(fpath);
            if (!f) return;
            std::string line;
            int lineno = 0;
            while (std::getline(f, line) && total < SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) {
                lineno++;
                if (std::regex_search(line, pattern)) {
                    output_text << fpath.string() << ":";
                    if (show_lineno) {
                        output_text << lineno << ":";
                    }
                    output_text << line << "\n";
                    total++;
                }
            }
        };

        std::error_code ec;
        if (fs::is_regular_file(path, ec)) {
            search_file(path);
        } else if (fs::is_directory(path, ec)) {
            for (const auto & entry : fs::recursive_directory_iterator(path,
                    fs::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) continue;
                if (total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) break;

                std::string rel = fs::relative(entry.path(), path, ec).string();
                if (ec) continue;
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(include, rel)) continue;
                if (!exclude.empty() && glob_match(exclude, rel)) continue;

                search_file(entry.path());
            }
        } else {
            return {{"error", "path does not exist: " + path}};
        }

        output_text << "\n\n---\nTotal matches: " << total << "\n";

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// exec_shell_command: run an arbitrary shell command
//

static constexpr size_t SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE = 16 * 1024; // 16 KB
static constexpr int    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT     = 60;        // seconds

struct server_tool_exec_shell_command : server_tool {
    server_tool_exec_shell_command() {
        name = "exec_shell_command";
        display_name = "Execute shell command";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Execute a shell command and return its output (stdout and stderr combined)."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"command",         {{"type", "string"},  {"description", "Shell command to execute"}}},
                        {"timeout",         {{"type", "integer"}, {"description", string_format("Timeout in seconds (default 10, max %d)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT)}}},
                        {"max_output_size", {{"type", "integer"}, {"description", string_format("Maximum output size in bytes (default %zu)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE)}}},
                    }},
                    {"required", json::array({"command"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string command   = params.at("command").get<std::string>();
        int    timeout        = json_value(params, "timeout",         10);
        size_t max_output     = (size_t) json_value(params, "max_output_size", (int) SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

        timeout    = std::min(timeout,    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT);
        max_output = std::min(max_output, SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

#ifdef _WIN32
        std::vector<std::string> args = {"cmd", "/c", command};
#else
        std::vector<std::string> args = {"sh", "-c", command};
#endif

        auto res = run_process(args, max_output, timeout);

        std::string text_output = res.output;
        text_output += string_format("\n[exit code: %d]", res.exit_code);
        if (res.timed_out) {
            text_output += " [exit due to timed out]";
        }

        return {{"plain_text_response", text_output}};
    }
};

//
// write_file: create or overwrite a file
//

struct server_tool_write_file : server_tool {
    server_tool_write_file() {
        name = "write_file";
        display_name = "Write file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Write content to a file, creating it (including parent directories) if it does not exist. May use with edit_file for more complex edits."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path of the file to write"}}},
                        {"content", {{"type", "string"}, {"description", "Content to write"}}},
                    }},
                    {"required", json::array({"path", "content"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string content = params.at("content").get<std::string>();

        std::error_code ec;
        fs::path fpath(path);
        if (fpath.has_parent_path()) {
            fs::create_directories(fpath.parent_path(), ec);
            if (ec) {
                return {{"error", "failed to create directories: " + ec.message()}};
            }
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        f << content;
        if (!f) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file written successfully"}, {"path", path}, {"bytes", content.size()}};
    }
};

//
// edit_file: edit file content via line-based changes
//

struct server_tool_edit_file : server_tool {
    server_tool_edit_file() {
        name = "edit_file";
        display_name = "Edit file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Edit a file by applying a list of line-based changes. "
                    "Each change targets a 1-based inclusive line range and has a mode: "
                    "\"replace\" (replace lines with content), "
                    "\"delete\" (remove lines, content must be empty string), "
                    "\"append\" (insert content after line_end). "
                    "Set line_start to -1 to target the end of file (line_end is ignored in that case). "
                    "Changes must not overlap. They are applied in reverse line order automatically."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path to the file to edit"}}},
                        {"changes", {
                            {"type", "array"},
                            {"description", "List of changes to apply"},
                            {"items", {
                                {"type", "object"},
                                {"properties", {
                                    {"mode",       {{"type", "string"},  {"description", "\"replace\", \"delete\", or \"append\""}}},
                                    {"line_start", {{"type", "integer"}, {"description", "First line of the range (1-based); use -1 for end of file"}}},
                                    {"line_end",   {{"type", "integer"}, {"description", "Last line of the range (1-based, inclusive); ignored when line_start is -1"}}},
                                    {"content",    {{"type", "string"},  {"description", "Content to insert; must be empty string for delete mode"}}},
                                }},
                                {"required", json::array({"mode", "line_start", "line_end", "content"})},
                            }},
                        }},
                    }},
                    {"required", json::array({"path", "changes"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path = params.at("path").get<std::string>();
        const json & changes = params.at("changes");

        if (!changes.is_array()) {
            return {{"error", "\"changes\" must be an array"}};
        }

        // read file into lines
        std::ifstream fin(path);
        if (!fin) {
            return {{"error", "failed to open file: " + path}};
        }
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(fin, line)) {
                lines.push_back(line);
            }
        }
        fin.close();

        // validate and collect changes, then sort descending by line_start
        struct change_entry {
            std::string mode;
            int line_start; // 1-based
            int line_end;   // 1-based inclusive
            std::string content;
        };
        std::vector<change_entry> entries;
        entries.reserve(changes.size());

        for (const auto & ch : changes) {
            change_entry e;
            e.mode       = ch.at("mode").get<std::string>();
            e.line_start = ch.at("line_start").get<int>();
            e.line_end   = ch.at("line_end").get<int>();
            e.content    = ch.at("content").get<std::string>();

            if (e.mode != "replace" && e.mode != "delete" && e.mode != "append") {
                return {{"error", "invalid mode \"" + e.mode + "\"; must be replace, delete, or append"}};
            }
            if (e.mode == "delete" && !e.content.empty()) {
                return {{"error", "content must be empty string for delete mode"}};
            }
            int n = (int) lines.size();
            if (e.line_start == -1) {
                // -1 means end of file; line_end is ignored — normalize to point past last line
                e.line_start = n + 1;
                e.line_end   = n + 1;
            } else {
                if (e.line_start < 1 || e.line_end < e.line_start) {
                    return {{"error", string_format("invalid line range [%d, %d]", e.line_start, e.line_end)}};
                }
                if (e.line_end > n) {
                    return {{"error", string_format("line_end %d exceeds file length %d", e.line_end, n)}};
                }
            }
            entries.push_back(std::move(e));
        }

        // sort descending so earlier-indexed changes don't shift later ones
        std::sort(entries.begin(), entries.end(), [](const change_entry & a, const change_entry & b) {
            return a.line_start > b.line_start;
        });

        // apply changes (0-based indices internally)
        for (const auto & e : entries) {
            int idx_start = e.line_start - 1; // 0-based
            int idx_end   = e.line_end   - 1; // 0-based inclusive

            // split content into lines (preserve trailing newline awareness)
            std::vector<std::string> new_lines;
            if (!e.content.empty()) {
                std::istringstream ss(e.content);
                std::string ln;
                while (std::getline(ss, ln)) {
                    new_lines.push_back(ln);
                }
                // if content ends with \n, getline consumed it — no extra empty line needed
                // if content does NOT end with \n, last line is still captured correctly
            }

            if (e.mode == "replace") {
                // erase [idx_start, idx_end] and insert new_lines
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
                lines.insert(lines.begin() + idx_start, new_lines.begin(), new_lines.end());
            } else if (e.mode == "delete") {
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
            } else { // append
                // idx_end + 1 may equal lines.size() when line_start == -1 (end of file)
                lines.insert(lines.begin() + idx_end + 1, new_lines.begin(), new_lines.end());
            }
        }

        // write file back
        std::ofstream fout(path, std::ios::binary);
        if (!fout) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        for (size_t i = 0; i < lines.size(); i++) {
            fout << lines[i];
            if (i + 1 < lines.size()) {
                fout << "\n";
            }
        }
        if (!lines.empty()) {
            fout << "\n";
        }
        if (!fout) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file edited successfully"}, {"path", path}, {"lines", (int) lines.size()}};
    }
};

//
// apply_diff: apply a unified diff via git apply
//

struct server_tool_apply_diff : server_tool {
    server_tool_apply_diff() {
        name = "apply_diff";
        display_name = "Apply diff";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Apply a unified diff to edit one or more files using git apply. Use this instead of edit_file when the changes are complex."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"diff", {{"type", "string"}, {"description", "Unified diff content in git diff format"}}},
                    }},
                    {"required", json::array({"diff"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string diff = params.at("diff").get<std::string>();

        // write diff to a temporary file
        static std::atomic<int> counter{0};
        std::string tmp_path = (fs::temp_directory_path() /
            ("llama_patch_" + std::to_string(++counter) + ".patch")).string();

        {
            std::ofstream f(tmp_path, std::ios::binary);
            if (!f) {
                return {{"error", "failed to create temp patch file"}};
            }
            f << diff;
        }

        auto res = run_process({"git", "apply", tmp_path}, 4096, 10);

        std::error_code ec;
        fs::remove(tmp_path, ec);

        if (res.exit_code != 0) {
            return {{"error", "git apply failed (exit " + std::to_string(res.exit_code) + "): " + res.output}};
        }
        return {{"result", "patch applied successfully"}};
    }
};

//
// public API
//

//
// search_code: BM25 + NER multihop search on indexed corpus
//

struct server_tool_search_code : server_tool {
    corpus_tool_context corpus;

    server_tool_search_code(const corpus_tool_context & ctx) : corpus(ctx) {
        name = "search_code";
        display_name = "Search code";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Search the indexed codebase for files matching a query. "
                                "Uses BM25 keyword matching + NER entity expansion for cross-file dependencies."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"query", {{"type", "string"}, {"description", "Search query (keywords, function names, class names, error messages)"}}},
                        {"k",     {{"type", "integer"}, {"description", "Maximum number of results (default: 10)"}}},
                    }},
                    {"required", json::array({"query"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        if (!corpus.bm25 || corpus.bm25->N == 0) {
            return {{"error", "no corpus indexed. Start server with --corpus-dir"}};
        }
        std::string query = params.at("query").get<std::string>();
        int k = json_value(params, "k", 10);

        auto hits = corpus.bm25->top_k(query, k);

        // Multihop expansion if entity index available
        {
            auto lk = corpus_lock(corpus);
            if (corpus.entities && corpus.linker && corpus.extractor && corpus.entities->n_unique > 0) {
                auto mh = multihop_query(query, hits,
                    *corpus.entities, *corpus.linker, *corpus.extractor, k);
                hits = mh.chunk_ids;
            }
        }

        json results = json::array();
        for (int cid : hits) {
            std::string source = corpus.bm25->get_source(cid);
            std::string text = corpus.bm25->get_text(cid);
            if (!text.empty()) {
                // Truncate to first 500 chars for summary
                std::string preview = text.substr(0, 500);
                if (text.size() > 500) preview += "...";
                results.push_back({
                    {"file", source.empty() ? "chunk_" + std::to_string(cid) : source},
                    {"chunk_id", cid},
                    {"preview", preview},
                });
            }
        }
        return {{"results", results}, {"total", (int)results.size()}};
    }
};

//
// trace_dependencies: NER import extraction + entity linking
//

struct server_tool_trace_dependencies : server_tool {
    corpus_tool_context corpus;

    server_tool_trace_dependencies(const corpus_tool_context & ctx) : corpus(ctx) {
        name = "trace_dependencies";
        display_name = "Trace dependencies";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Given a source file, find all files it imports or depends on "
                                "via import/include/require statements and entity linking. "
                                "Cost: instant (no LLM call, pure regex + index lookup)."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"file", {{"type", "string"}, {"description", "Source filename to trace dependencies for"}}},
                    }},
                    {"required", json::array({"file"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        if (!corpus.bm25 || corpus.bm25->N == 0) {
            return {{"error", "no corpus indexed"}};
        }
        std::string file = params.at("file").get<std::string>();

        // Find the chunk for this file
        std::string text;
        for (int i = 0; i < corpus.bm25->N; i++) {
            const int cid = corpus.bm25->chunk_id_at((size_t) i);   // no reference into docs[]: adds may run
            if (cid < 0) break;
            std::string src = corpus.bm25->get_source(cid);
            if (src == file || src.find(file) != std::string::npos) {
                text = corpus.bm25->get_text(cid);
                break;
            }
        }

        if (text.empty()) {
            return {{"error", "file not found in corpus: " + file}};
        }

        auto lk = corpus_lock(corpus);

        // Extract imports using NER
        json deps = json::array();
        if (corpus.extractor) {
            auto imports = corpus.extractor->extract_imports(text);
            for (auto & imp : imports) {
                deps.push_back(imp);
            }
        }

        // Entity linker resolution
        json linked = json::array();
        if (corpus.linker) {
            auto resolved = corpus.linker->resolve(file);
            for (auto & r : resolved) {
                linked.push_back(r);
            }
        }

        return {{"file", file}, {"imports", deps}, {"linked_entities", linked}};
    }
};

//
// read_corpus_file: read a file from the indexed corpus (not filesystem)
//

struct server_tool_read_corpus_file : server_tool {
    corpus_tool_context corpus;

    server_tool_read_corpus_file(const corpus_tool_context & ctx) : corpus(ctx) {
        name = "read_corpus_file";
        display_name = "Read corpus file";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read the full contents of a file from the indexed corpus. "
                                "Use this after search_code to read a specific file's code."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"file", {{"type", "string"}, {"description", "Filename or path to read from the indexed corpus"}}},
                    }},
                    {"required", json::array({"file"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        if (!corpus.bm25 || corpus.bm25->N == 0) {
            return {{"error", "no corpus indexed"}};
        }
        std::string file = params.at("file").get<std::string>();

        for (int i = 0; i < corpus.bm25->N; i++) {
            const int cid = corpus.bm25->chunk_id_at((size_t) i);   // no reference into docs[]: adds may run
            if (cid < 0) break;
            std::string src = corpus.bm25->get_source(cid);
            if (src == file || src.find(file) != std::string::npos) {
                std::string text = corpus.bm25->get_text(cid);
                return {{"file", src}, {"content", text}};
            }
        }
        return {{"error", "file not found in corpus: " + file}};
    }
};

//
// Experience system: learned lessons stored in the corpus and retrieved for later tasks
//
// 8 experience types: knowledge, bugfix, workflow, mistake,
//                     preference, calibration, environment, recovery
//

// Sanitize user-provided text before storing in experiences.
// Strips prompt injection patterns, special tokens, and role-switching.
static std::string sanitize_experience_text(const std::string & text) {
    std::string s = text;
    // Strip special tokens that could affect model behavior
    static const std::vector<std::string> bad_tokens = {
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>",
        "</system>", "<system>", "</user>", "<assistant>",
        "<|system|>", "<|user|>", "<|assistant|>",
    };
    for (auto & tok : bad_tokens) {
        size_t pos;
        while ((pos = s.find(tok)) != std::string::npos) {
            s.erase(pos, tok.size());
        }
    }
    // Strip lines that look like role-switching instructions
    // (e.g., "NEW INSTRUCTION:", "IGNORE ALL PREVIOUS", "You are now")
    static const std::regex re_injection(
        R"((?:^|\n)\s*(?:ignore\s+all|disregard|new\s+instruction|you\s+are\s+now|system\s*:)[^\n]*)",
        std::regex::icase);
    s = std::regex_replace(s, re_injection, "");
    // Limit length
    if (s.size() > 2000) s.resize(2000);
    return s;
}

struct experience_workflow {
    std::string plan_target;
    std::string plan_description;
    struct action { std::string file; std::string description; };
    struct test_result { std::string command; int exit_code; std::string output; };
    std::vector<action> actions;
    std::vector<test_result> tests;
    bool submitted = false;
    int next_exp_id = 100000;  // experience chunk IDs start high to avoid collision

    bool has_plan() const { return !plan_target.empty(); }
    bool has_actions() const { return !actions.empty(); }
    bool last_test_passed() const { return !tests.empty() && tests.back().exit_code == 0; }
    bool can_submit() const { return has_plan() && has_actions() && last_test_passed() && !submitted; }

    // Determine verification status for auto-extraction
    // verified: plan + fix + test passed + submitted
    // attempted: plan + fix, but no test or test failed
    // observation: plan only, no fix
    std::string auto_status() const {
        if (submitted) return "verified";
        if (has_actions()) return "attempted";
        if (has_plan()) return "observation";
        return "";
    }

    bool can_auto_extract() const {
        return has_plan() && !submitted;
    }

    void reset() {
        plan_target.clear(); plan_description.clear();
        actions.clear(); tests.clear(); submitted = false;
    }

    json to_json() const {
        json j;
        j["plan"] = has_plan() ? json{{"target", plan_target}, {"description", plan_description}} : json(nullptr);
        json fixes = json::array();
        for (auto & a : actions) fixes.push_back({{"file", a.file}, {"description", a.description}});
        j["fixes"] = fixes;
        json test_arr = json::array();
        for (auto & t : tests) test_arr.push_back({{"command", t.command}, {"exit_code", t.exit_code}, {"result", t.exit_code == 0 ? "PASS" : "FAIL"}});
        j["tests"] = test_arr;
        j["submitted"] = submitted;
        j["ready_to_submit"] = can_submit();
        if (!can_submit() && has_plan()) {
            if (!has_actions()) j["blocked_reason"] = "no fixes recorded";
            else if (tests.empty()) j["blocked_reason"] = "no tests run";
            else if (!last_test_passed()) j["blocked_reason"] = "last test failed";
            else if (submitted) j["blocked_reason"] = "already submitted";
        }
        return j;
    }
};

// Shared state - one workflow per server instance
static experience_workflow g_experience;

// Build experience text from current workflow state.
// status: "verified", "attempted", or "observation"
// model_name: for model-specific tagging
static std::string build_experience_text(const std::string & status, const std::string & model_name) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    char date_buf[32];
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", std::localtime(&tt));

    // Determine type
    std::string exp_type;
    if (status == "observation") {
        exp_type = "observation";
    } else {
        bool had_failure = false;
        for (auto & t : g_experience.tests) {
            if (t.exit_code != 0) had_failure = true;
        }
        exp_type = had_failure ? "bugfix" : "knowledge";
    }

    // Build context
    std::string context = "Target: " + g_experience.plan_target + "\n"
                        + "Problem: " + g_experience.plan_description + "\n";
    for (auto & a : g_experience.actions) {
        std::string prefix = (status == "attempted") ? "Attempted fix" : "Fix";
        context += prefix + " [" + a.file + "]: " + a.description + "\n";
    }

    // Build lesson based on status
    std::string lesson;
    if (status == "observation") {
        lesson = "OBSERVATION: Identified problem in " + g_experience.plan_target
               + " — " + g_experience.plan_description
               + ". No fix attempted (timeout/stuck).";
    } else if (status == "attempted") {
        // Attempted but not verified
        bool had_test_fail = false;
        std::string last_err;
        for (auto & t : g_experience.tests) {
            if (t.exit_code != 0) {
                had_test_fail = true;
                auto nl = t.output.find('\n');
                last_err = t.output.substr(0, std::min(nl, (size_t)200));
            }
        }
        lesson = "ATTEMPTED: ";
        if (!g_experience.actions.empty()) {
            lesson += g_experience.actions.back().description;
        }
        if (had_test_fail && !last_err.empty()) {
            lesson += ". TEST FAILED: " + last_err;
        }
        lesson += ". NOT VERIFIED. Use with caution — fix was never test-confirmed.";
    } else {
        // Verified
        bool had_failure = false;
        for (auto & t : g_experience.tests) {
            if (t.exit_code != 0) had_failure = true;
        }
        if (had_failure) {
            std::vector<std::string> failed_attempts;
            std::string successful_fix;
            for (size_t i = 0; i < g_experience.tests.size(); i++) {
                if (g_experience.tests[i].exit_code != 0) {
                    auto & out = g_experience.tests[i].output;
                    auto nl = out.find('\n');
                    failed_attempts.push_back(out.substr(0, std::min(nl, (size_t)200)));
                }
            }
            if (!g_experience.actions.empty()) {
                auto & last = g_experience.actions.back();
                successful_fix = last.file + ": " + last.description;
            }
            lesson = "MISTAKE: ";
            if (!failed_attempts.empty()) {
                lesson += "First approach failed with: " + failed_attempts[0] + ". ";
            }
            if (failed_attempts.size() > 1) {
                lesson += std::to_string(failed_attempts.size()) + " attempts total. ";
            }
            if (!successful_fix.empty()) {
                lesson += "FIX THAT WORKED: " + successful_fix + ". ";
            }
            lesson += "PATTERN: test after each change, iterate on failures.";
        } else {
            if (!g_experience.actions.empty()) {
                auto & a = g_experience.actions[0];
                lesson = "SOLUTION: " + a.description;
                if (g_experience.actions.size() > 1) {
                    lesson += " (+ " + std::to_string(g_experience.actions.size() - 1) + " more changes)";
                }
                lesson += ". Files: ";
                for (size_t i = 0; i < g_experience.actions.size(); i++) {
                    if (i > 0) lesson += ", ";
                    lesson += g_experience.actions[i].file;
                }
                lesson += ".";
            } else {
                lesson = g_experience.plan_description;
            }
        }
    }

    // Build tags
    std::string tags;
    for (auto & a : g_experience.actions) {
        auto dot = a.file.find_last_of('.');
        if (dot != std::string::npos) {
            std::string ext = a.file.substr(dot + 1);
            if (tags.find(ext) == std::string::npos) {
                if (!tags.empty()) tags += " ";
                tags += ext;
            }
        }
        auto slash = a.file.find_last_of("/\\");
        std::string base = (slash != std::string::npos) ? a.file.substr(slash + 1) : a.file;
        if (!tags.empty()) tags += " ";
        tags += base;
    }

    // Model tag
    static const std::unordered_set<std::string> model_specific_types = {
        "mistake", "calibration", "recovery",
    };
    std::string model_tag = model_specific_types.count(exp_type)
        ? model_name : "any";

    return "[Experience: " + exp_type + " | " + date_buf + "]\n"
         + "[Model: " + model_tag + "]\n"
         + "[Status: " + status + "]\n"
         + "[Source: local]\n"
         + "[Context]: " + context + "\n"
         + "[Lesson]: " + lesson + "\n"
         + "[Tags]: " + tags + "\n";
}

// Save experience to BM25 + disk. Returns exp_id.
static int save_experience(const std::string & text, bm25_index * bm25, const std::string & corpus_dir) {
    int exp_id = g_experience.next_exp_id++;
    std::string exp_source = "experience/exp_" + std::to_string(exp_id);
    if (bm25) {
        bm25->add_document(exp_id, text, exp_source);
    }
    if (!corpus_dir.empty()) {
        namespace fs = std::filesystem;
        std::string exp_dir = corpus_dir + "/experiences";
        std::error_code ec;
        fs::create_directories(exp_dir, ec);
        std::string exp_path = exp_dir + "/exp_" + std::to_string(exp_id) + ".md";
        std::ofstream f(exp_path);
        if (f) { f << text; f.close(); }
    }
    return exp_id;
}

struct server_tool_experience_plan : server_tool {
    corpus_tool_context corpus;
    std::string corpus_dir;

    server_tool_experience_plan(const corpus_tool_context & ctx = {}, const std::string & dir = "")
        : corpus(ctx), corpus_dir(dir) {
        name = "experience_plan";
        display_name = "Experience: Plan";
        permission_write = false;
    }
    json get_definition() override {
        return {{"type", "function"}, {"function", {
            {"name", name},
            {"description", "Start a new workflow (bug fix, investigation, learning). Resets any previous state."},
            {"parameters", {{"type", "object"}, {"properties", {
                {"target", {{"type", "string"}, {"description", "Target file or topic"}}},
                {"description", {{"type", "string"}, {"description", "What needs to be done"}}},
            }}, {"required", json::array({"target", "description"})}}},
        }}};
    }
    json invoke(json params) override {
        bool had_previous = g_experience.has_plan();
        std::string prev_target = g_experience.plan_target;

        // Auto-extract partial experience from unsubmitted workflow
        int auto_id = -1;
        std::string auto_status;
        if (had_previous && g_experience.can_auto_extract()) {
            auto_status = g_experience.auto_status();
            std::string text = build_experience_text(auto_status, corpus.model_name);
            auto_id = save_experience(text, corpus.bm25, corpus_dir);
        }

        g_experience.reset();
        g_experience.plan_target = sanitize_experience_text(params.at("target").get<std::string>());
        g_experience.plan_description = sanitize_experience_text(params.at("description").get<std::string>());
        json result = {{"status", "planned"}, {"target", g_experience.plan_target}, {"description", g_experience.plan_description}};
        if (had_previous) {
            result["warning"] = "previous plan for '" + prev_target + "' was overwritten";
            if (auto_id >= 0) {
                result["auto_extracted"] = auto_id;
                result["auto_extracted_status"] = auto_status;
            }
        }
        return result;
    }
};

struct server_tool_experience_fix : server_tool {
    server_tool_experience_fix() {
        name = "experience_fix";
        display_name = "Experience: Fix";
        permission_write = false;
    }
    json get_definition() override {
        return {{"type", "function"}, {"function", {
            {"name", name},
            {"description", "Record an action/fix taken. Must call experience_plan first."},
            {"parameters", {{"type", "object"}, {"properties", {
                {"file", {{"type", "string"}, {"description", "File that was modified"}}},
                {"description", {{"type", "string"}, {"description", "What was changed"}}},
            }}, {"required", json::array({"file", "description"})}}},
        }}};
    }
    json invoke(json params) override {
        if (!g_experience.has_plan()) {
            return {{"error", "no plan set. Call experience_plan first."}};
        }
        g_experience.actions.push_back({
            sanitize_experience_text(params.at("file").get<std::string>()),
            sanitize_experience_text(params.at("description").get<std::string>()),
        });
        return {{"status", "recorded"}, {"n_fixes", (int)g_experience.actions.size()}};
    }
};

static constexpr int    EXPERIENCE_TEST_TIMEOUT    = 60;
static constexpr size_t EXPERIENCE_TEST_MAX_OUTPUT = 16 * 1024;

struct server_tool_experience_test : server_tool {
    server_tool_experience_test() {
        name = "experience_test";
        display_name = "Experience: Test";
        permission_write = false;
    }
    json get_definition() override {
        return {{"type", "function"}, {"function", {
            {"name", name},
            {"description", "Execute a test command. Records PASS/FAIL. Must have at least one fix recorded."},
            {"parameters", {{"type", "object"}, {"properties", {
                {"command", {{"type", "string"}, {"description", "Shell command to run as test"}}},
            }}, {"required", json::array({"command"})}}},
        }}};
    }
    json invoke(json params) override {
        if (!g_experience.has_plan()) {
            return {{"error", "no plan set. Call experience_plan first."}};
        }
        if (!g_experience.has_actions()) {
            return {{"error", "no fixes recorded. Call experience_fix first."}};
        }
        std::string command = params.at("command").get<std::string>();
#ifdef _WIN32
        std::vector<std::string> args = {"cmd", "/c", command};
#else
        std::vector<std::string> args = {"sh", "-c", command};
#endif
        auto res = run_process(args, EXPERIENCE_TEST_MAX_OUTPUT, EXPERIENCE_TEST_TIMEOUT);
        g_experience.tests.push_back({command, res.exit_code, res.output});

        std::string result_str = res.exit_code == 0 ? "PASS" : "FAIL";
        json r = {{"result", result_str}, {"exit_code", res.exit_code}, {"output", res.output}};
        if (res.timed_out) r["timed_out"] = true;
        return r;
    }
};

struct server_tool_experience_status : server_tool {
    server_tool_experience_status() {
        name = "experience_status";
        display_name = "Experience: Status";
        permission_write = false;
    }
    json get_definition() override {
        return {{"type", "function"}, {"function", {
            {"name", name},
            {"description", "Show current workflow state: plan, fixes, tests, ready-to-submit status."},
            {"parameters", {{"type", "object"}, {"properties", json::object()}}},
        }}};
    }
    json invoke(json) override {
        return g_experience.to_json();
    }
};

struct server_tool_experience_submit : server_tool {
    corpus_tool_context corpus;
    std::string corpus_dir;

    server_tool_experience_submit(const corpus_tool_context & ctx, const std::string & dir)
        : corpus(ctx), corpus_dir(dir) {
        name = "experience_submit";
        display_name = "Experience: Submit";
        permission_write = false;
    }
    json get_definition() override {
        return {{"type", "function"}, {"function", {
            {"name", name},
            {"description", "Complete the workflow and extract an Experience. "
                            "BLOCKED unless: plan exists, at least one fix recorded, and last test passed."},
            {"parameters", {{"type", "object"}, {"properties", json::object()}}},
        }}};
    }
    json invoke(json) override {
        if (!g_experience.can_submit()) {
            json status = g_experience.to_json();
            return {{"error", "cannot submit"}, {"reason", status.value("blocked_reason", "unknown")}, {"status", status}};
        }

        std::string experience_text = build_experience_text("verified", corpus.model_name);
        int exp_id = save_experience(experience_text, corpus.bm25, corpus_dir);
        g_experience.submitted = true;

        return {{"status", "submitted"}, {"experience_id", exp_id},
                {"experience", experience_text}};
    }
};

// Harness calls this on timeout to extract partial experience from unfinished workflow.
// Not a model tool - called by external harness via /tools endpoint.
struct server_tool_experience_extract : server_tool {
    corpus_tool_context corpus;
    std::string corpus_dir;

    server_tool_experience_extract(const corpus_tool_context & ctx, const std::string & dir)
        : corpus(ctx), corpus_dir(dir) {
        name = "experience_extract";
        display_name = "Experience: Auto-Extract";
        permission_write = false;
    }
    json get_definition() override {
        return {{"type", "function"}, {"function", {
            {"name", name},
            {"description", "Auto-extract partial experience from unfinished workflow (for harness timeout). "
                            "Extracts whatever plan/fix/test state exists without requiring submit conditions."},
            {"parameters", {{"type", "object"}, {"properties", json::object()}}},
        }}};
    }
    json invoke(json) override {
        if (!g_experience.can_auto_extract()) {
            return {{"status", "nothing_to_extract"}, {"reason", "no plan set or already submitted"}};
        }
        std::string status = g_experience.auto_status();
        std::string text = build_experience_text(status, corpus.model_name);
        int exp_id = save_experience(text, corpus.bm25, corpus_dir);
        g_experience.submitted = true;  // prevent double-extraction

        return {{"status", "auto_extracted"}, {"experience_id", exp_id},
                {"verification_status", status}, {"experience", text}};
    }
};

struct server_tool_experience_search : server_tool {
    corpus_tool_context corpus;

    server_tool_experience_search(const corpus_tool_context & ctx) : corpus(ctx) {
        name = "experience_search";
        display_name = "Experience: Search";
        permission_write = false;
    }
    json get_definition() override {
        return {{"type", "function"}, {"function", {
            {"name", name},
            {"description", "Search past experiences by keyword. Returns relevant lessons from previous sessions."},
            {"parameters", {{"type", "object"}, {"properties", {
                {"query", {{"type", "string"}, {"description", "Search keywords"}}},
                {"k", {{"type", "integer"}, {"description", "Max results (default: 5)"}}},
            }}, {"required", json::array({"query"})}}},
        }}};
    }
    json invoke(json params) override {
        if (!corpus.bm25) return {{"error", "no corpus indexed"}};
        std::string query = params.at("query").get<std::string>();
        int k = json_value(params, "k", 5);

        auto hits = corpus.bm25->top_k(query, k * 5);  // over-fetch then filter
        json results = json::array();
        for (int cid : hits) {
            std::string source = corpus.bm25->get_source(cid);
            if (source.find("experience") == 0) {  // matches experience/ and experiences/
                std::string text = corpus.bm25->get_text(cid);

                // Filter model-specific experiences: skip if tagged for a different model
                // [Model: any] -> always include
                // [Model: qwen3.6-35b] -> include only if current model matches
                auto model_pos = text.find("[Model: ");
                if (model_pos != std::string::npos) {
                    auto model_end = text.find("]", model_pos + 8);
                    if (model_end != std::string::npos) {
                        std::string tagged_model = text.substr(model_pos + 8, model_end - model_pos - 8);
                        if (tagged_model != "any" && !corpus.model_name.empty()) {
                            // Model-specific: skip if different model
                            if (tagged_model.find(corpus.model_name) == std::string::npos &&
                                corpus.model_name.find(tagged_model) == std::string::npos) {
                                continue;  // wrong model, skip this experience
                            }
                        }
                    }
                }

                results.push_back({{"id", cid}, {"source", source}, {"content", text}});
                if ((int)results.size() >= k) break;
            }
        }
        return {{"results", results}, {"total", (int)results.size()}, {"model", corpus.model_name}};
    }
};

struct server_tool_experience_load : server_tool {
    corpus_tool_context corpus;

    server_tool_experience_load(const corpus_tool_context & ctx) : corpus(ctx) {
        name = "experience_load";
        display_name = "Experience: Load";
        permission_write = false;
    }
    json get_definition() override {
        return {{"type", "function"}, {"function", {
            {"name", name},
            {"description", "Load an external experience file into the corpus. "
                            "Use for experiences downloaded from alphachat.com or shared by other users."},
            {"parameters", {{"type", "object"}, {"properties", {
                {"content", {{"type", "string"}, {"description", "Experience text in standard format"}}},
                {"source", {{"type", "string"}, {"description", "Source identifier (e.g. alphachat.com/experiences/...)"}}},
            }}, {"required", json::array({"content"})}}},
        }}};
    }
    json invoke(json params) override {
        if (!corpus.bm25) return {{"error", "no corpus indexed"}};
        std::string content = sanitize_experience_text(params.at("content").get<std::string>());
        std::string source = json_value(params, "source", std::string("experience/external"));

        int exp_id = g_experience.next_exp_id++;
        std::string src = "experience/external/" + std::to_string(exp_id);
        corpus.bm25->add_document(exp_id, content, src);

        return {{"status", "loaded"}, {"experience_id", exp_id}, {"source", src}};
    }
};

static std::vector<std::unique_ptr<server_tool>> build_tools(
        const corpus_tool_context & corpus_ctx = {},
        const std::string & corpus_dir = "") {
    std::vector<std::unique_ptr<server_tool>> tools;
    tools.push_back(std::make_unique<server_tool_read_file>());
    tools.push_back(std::make_unique<server_tool_file_glob_search>());
    tools.push_back(std::make_unique<server_tool_grep_search>());
    tools.push_back(std::make_unique<server_tool_exec_shell_command>());
    tools.push_back(std::make_unique<server_tool_write_file>());
    tools.push_back(std::make_unique<server_tool_edit_file>());
    tools.push_back(std::make_unique<server_tool_apply_diff>());
    // Corpus-aware tools (only when corpus is indexed)
    if (corpus_ctx.bm25 && corpus_ctx.bm25->N > 0) {
        tools.push_back(std::make_unique<server_tool_search_code>(corpus_ctx));
        tools.push_back(std::make_unique<server_tool_trace_dependencies>(corpus_ctx));
        tools.push_back(std::make_unique<server_tool_read_corpus_file>(corpus_ctx));
    }
    // Experience tools (always available when corpus exists)
    if (corpus_ctx.bm25) {
        // Scan existing experience files to avoid ID collision on restart
        if (!corpus_dir.empty()) {
            namespace fs = std::filesystem;
            std::string exp_dir = corpus_dir + "/experiences";
            std::error_code ec;
            if (fs::is_directory(exp_dir, ec)) {
                int max_id = 99999;
                for (auto & entry : fs::directory_iterator(exp_dir, ec)) {
                    auto fname = entry.path().stem().string(); // "exp_100003"
                    auto pos = fname.find('_');
                    if (pos != std::string::npos) {
                        try { max_id = std::max(max_id, std::stoi(fname.substr(pos + 1))); }
                        catch (...) {}
                    }
                }
                g_experience.next_exp_id = max_id + 1;
            }
        }
        tools.push_back(std::make_unique<server_tool_experience_plan>(corpus_ctx, corpus_dir));
        tools.push_back(std::make_unique<server_tool_experience_fix>());
        tools.push_back(std::make_unique<server_tool_experience_test>());
        tools.push_back(std::make_unique<server_tool_experience_status>());
        tools.push_back(std::make_unique<server_tool_experience_submit>(corpus_ctx, corpus_dir));
        tools.push_back(std::make_unique<server_tool_experience_search>(corpus_ctx));
        tools.push_back(std::make_unique<server_tool_experience_load>(corpus_ctx));
        tools.push_back(std::make_unique<server_tool_experience_extract>(corpus_ctx, corpus_dir));
    }
    return tools;
}

void server_tools::setup(const std::vector<std::string> & enabled_tools,
                         const corpus_tool_context & corpus_ctx) {
    if (!enabled_tools.empty()) {
        std::unordered_set<std::string> enabled_set(enabled_tools.begin(), enabled_tools.end());
        auto all_tools = build_tools(corpus_ctx, corpus_dir);

        // "all" = corpus tools only (safe). "all-unsafe" = everything including filesystem.
        bool enable_all_safe   = enabled_set.count("all") > 0;
        bool enable_all_unsafe = enabled_set.count("all-unsafe") > 0;

        static const std::unordered_set<std::string> corpus_tools = {
            "search_code", "trace_dependencies", "read_corpus_file",
            "experience_plan", "experience_fix",
            "experience_status", "experience_submit",
            "experience_search", "experience_load", "experience_extract",
            // experience_test is not here: it executes shell commands, so it belongs in
            // all-unsafe only.
        };

        tools.clear();
        for (auto & t : all_tools) {
            if (enable_all_unsafe) {
                // Enable everything - filesystem + corpus
                tools.push_back(std::move(t));
            } else if (enable_all_safe && corpus_tools.count(t->name) > 0) {
                // "all" = only corpus-safe tools
                tools.push_back(std::move(t));
            } else if (enabled_set.count(t->name) > 0) {
                // Explicitly named tool
                tools.push_back(std::move(t));
            }
        }
    }

    handle_get = [this](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json result = json::array();
            for (const auto & t : tools) {
                result.push_back(t->to_json());
            }
            res->data = safe_json_to_str(result);
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };

    handle_post = [this](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json body = json::parse(req.body);
            std::string tool_name = body.at("tool").get<std::string>();
            json params = body.value("params", json::object());
            json result = invoke(tool_name, params);
            res->data   = safe_json_to_str(result);
        } catch (const json::exception & e) {
            res->status = 400;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };
}

json server_tools::invoke(const std::string & name, const json & params) {
    for (auto & t : tools) {
        if (t->name == name) {
            return t->invoke(params);
        }
    }
    return {{"error", "unknown tool: " + name}};
}
