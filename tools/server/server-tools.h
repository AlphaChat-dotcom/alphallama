#pragma once

#include "server-common.h"
#include "server-http.h"

struct corpus_tool_context;  // forward declaration (defined in server-context.h)

struct server_tool {
    std::string name;
    std::string display_name;
    bool permission_write = false;

    virtual ~server_tool() = default;
    virtual json get_definition() = 0;
    virtual json invoke(json params) = 0;

    json to_json();
};

struct server_tools {
    std::vector<std::unique_ptr<server_tool>> tools;
    std::string corpus_dir;  // for experience persistence

    void setup(const std::vector<std::string> & enabled_tools,
               const corpus_tool_context & corpus_ctx);
    json invoke(const std::string & name, const json & params);

    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;
};
