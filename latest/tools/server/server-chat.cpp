#include "server-chat.h"
#include "server-common.h"

#include <sstream>

json server_chat_convert_responses_tools(const json & tools) {
    if (!tools.is_array()) {
        throw std::invalid_argument("'tools' must be an array of objects");
    }
    json chatcmpl_tools = json::array();
    for (json resp_tool : tools) {
        json chatcmpl_tool;

        const std::string type = json_value(resp_tool, "type", std::string());
        if (type != "function") {
            // Non-function Responses tools have no Chat Completions equivalent.
            SRV_WRN("unsupported Responses tool type '%s' skipped\n", type.c_str());
            continue;
        }
        resp_tool.erase("type");
        chatcmpl_tool["type"] = "function";

        if (!resp_tool.contains("strict")) {
            resp_tool["strict"] = true;
        }
        chatcmpl_tool["function"] = resp_tool;
        chatcmpl_tools.push_back(chatcmpl_tool);
    }
    return chatcmpl_tools;
}

json server_chat_convert_responses_to_chatcmpl(const json & response_body) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        throw std::invalid_argument("llama.cpp does not support 'previous_response_id'.");
    }

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    std::vector<json> chatcmpl_messages;

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        // #responses_create-input-text_input
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    } else if (input_value.is_array()) {
        // #responses_create-input-input_item_list

        static auto exists_and_is_array = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_array();
        };
        static auto exists_and_is_string = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_string();
        };

        for (json item : input_value) {
            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";

            if (exists_and_is_string(item, "content")) {
                // #responses_create-input-input_item_list-input_message-content-text_input
                // Only "Input message" contains item["content"]::string
                // After converting item["content"]::string to item["content"]::array,
                // we can treat "Input message" as sum of "Item-Input message" and "Item-Output message"
                item["content"] = json::array({
                    json {
                        {"text", item.at("content")},
                        {"type", "input_text"}
                    }
                });
            }

            if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                (item.at("role") == "user" ||
                    item.at("role") == "system" ||
                    item.at("role") == "developer")
            ) {
                // #responses_create-input-input_item_list-item-input_message
                std::vector<json> chatcmpl_content;

                for (const json & input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            throw std::invalid_argument("'Input text' requires 'text'");
                        }
                        chatcmpl_content.push_back({
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        });
                    } else if (type == "input_image") {
                        // While `detail` is marked as required,
                        // it has default value("auto") and can be omitted.

                        if (!input_item.contains("image_url")) {
                            throw std::invalid_argument("'image_url' is required");
                        }
                        chatcmpl_content.push_back({
                            {"image_url", json {
                                {"url", input_item.at("image_url")}
                            }},
                            {"type", "image_url"},
                        });
                    } else if (type == "input_file") {
                        throw std::invalid_argument("'input_file' is not supported by llamacpp at this moment");
                    } else {
                        throw std::invalid_argument("'type' must be one of 'input_text', 'input_image', or 'input_file'");
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            } else if (exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "message"
            ) {
                // #responses_create-input-input_item_list-item-output_message
                auto chatcmpl_content = json::array();

                // Handle both string content and array content
                if (item.contains("content") && item.at("content").is_string()) {
                    // String content - convert to text content part
                    chatcmpl_content.push_back({
                        {"text", item.at("content")},
                        {"type", "text"},
                    });
                } else if (exists_and_is_array(item, "content")) {
                    // Array content - process each item
                    for (const auto & output_text : item.at("content")) {
                        const std::string type = json_value(output_text, "type", std::string());
                        if (type == "output_text" || type == "input_text") {
                            // Accept both output_text and input_text (string content gets converted to input_text)
                            if (!exists_and_is_string(output_text, "text")) {
                                throw std::invalid_argument("'Output text' requires 'text'");
                            }
                            chatcmpl_content.push_back({
                                {"text", output_text.at("text")},
                                {"type", "text"},
                            });
                        } else if (type == "refusal") {
                            if (!exists_and_is_string(output_text, "refusal")) {
                                throw std::invalid_argument("'Refusal' requires 'refusal'");
                            }
                            chatcmpl_content.push_back({
                                {"refusal", output_text.at("refusal")},
                                {"type", "refusal"},
                            });
                        } else {
                            throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
                        }
                    }
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
                        prev_msg["content"] = json::array();
                    }
                    auto & prev_content = prev_msg["content"];
                    prev_content.insert(chatcmpl_content);
                } else {
                    item.erase("status");
                    item.erase("type");
                    item["content"] = chatcmpl_content;
                    chatcmpl_messages.push_back(item);
                }
            } else if (exists_and_is_string(item, "arguments") &&
                exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call
                json tool_call = {
                    {"function", json {
                        {"arguments", item.at("arguments")},
                        {"name",      item.at("name")},
                    }},
                    {"id",   item.at("call_id")},
                    {"type", "function"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "tool_calls")) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
            } else if (exists_and_is_string(item, "call_id") &&
                (exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call_output"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call_output
                if (item.at("output").is_string()) {
                    chatcmpl_messages.push_back(json {
                        {"content",      item.at("output")},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                } else {
                    json chatcmpl_outputs = item.at("output");
                    for (json & chatcmpl_output : chatcmpl_outputs) {
                        if (!chatcmpl_output.contains("type")) {
                            throw std::invalid_argument("Output of tool call missing 'type' field");
                        }
                        const auto type = chatcmpl_output.at("type");
                        if (type != "input_text" && type != "input_image") {
                            throw std::invalid_argument("Output of tool call should be 'Input text' or 'Input image'");
                        }
                        if (type == "input_text") {
                            chatcmpl_output["type"] = "text";
                        } else if (type == "input_image") {
                            if (!chatcmpl_output.contains("image_url")) {
                                throw std::invalid_argument("'image_url' is required");
                            }
                            chatcmpl_output["type"] = "image_url";
                            chatcmpl_output["image_url"] = json {
                                {"url", chatcmpl_output.at("image_url")}
                            };
                        }
                    }
                    chatcmpl_messages.push_back(json {
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
            } else if (exists_and_is_array(item, "summary") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                // #responses_create-input-input_item_list-item-reasoning

                if (!exists_and_is_array(item, "content")) {
                    throw std::invalid_argument("item['content'] is not an array");
                }
                if (item.at("content").empty()) {
                    throw std::invalid_argument("item['content'] is empty");
                }
                if (!exists_and_is_string(item.at("content")[0], "text")) {
                    throw std::invalid_argument("item['content']['text'] is not a string");
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    prev_msg["reasoning_content"] = item.at("content")[0].at("text");
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role", "assistant"},
                        {"content", json::array()},
                        {"reasoning_content", item.at("content")[0].at("text")},
                    });
                }
            } else {
                throw std::invalid_argument("Cannot determine type of 'item'");
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or array of objects");
    }

    chatcmpl_body["messages"] = chatcmpl_messages;

    if (response_body.contains("tools")) {
        const json chatcmpl_tools = server_chat_convert_responses_tools(response_body.at("tools"));
        chatcmpl_body.erase("tools");
        if (!chatcmpl_tools.empty()) {
            chatcmpl_body["tools"] = chatcmpl_tools;
        }
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    if (response_body.contains("reasoning")) {
        // Only "effort" is handled so far
        const json & reasoning = response_body.at("reasoning");
        if (reasoning.contains("effort")) {
            chatcmpl_body["reasoning_effort"] = reasoning.at("effort");
        }
        chatcmpl_body.erase("reasoning");
    }

    if (response_body.contains("text")) {
        const json & text = response_body.at("text");
        if (!text.is_object()) {
            throw std::invalid_argument("'text' must be an object");
        }

        chatcmpl_body.erase("text");
        if (text.contains("format")) {
            const json & format = text.at("format");
            if (!format.is_object()) {
                throw std::invalid_argument("'text.format' must be an object");
            }

            const std::string type = json_value(format, "type", std::string());
            if (type == "json_schema") {
                chatcmpl_body["response_format"] = {
                    {"type", "json_schema"},
                    {"json_schema", format},
                };
                chatcmpl_body["response_format"]["json_schema"].erase("type");
            } else if (type == "json_object" || type == "text") {
                chatcmpl_body["response_format"] = format;
            } else if (!type.empty()) {
                throw std::invalid_argument("'text.format.type' must be one of 'text', 'json_object', or 'json_schema'");
            }
        }
    }

    // llama.cpp params (e.g. seed, min_p) go under extra_params and are merged into the converted body
    if (response_body.contains("extra_params")) {
        chatcmpl_body.erase("extra_params");
        if (response_body.at("extra_params").is_object()) {
            for (const auto & [key, value] : response_body.at("extra_params").items()) {
                chatcmpl_body[key] = value;
            }
        }
    }

    return chatcmpl_body;
}

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"]   = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({ tool_call });
    }
    return delta;
}

json convert_transcriptions_to_chatcmpl(
        const json & inp_body,
        const common_chat_templates * tmpls,
        const std::map<std::string, uploaded_file> & in_files,
        std::vector<raw_buffer> & out_files) {
    // TODO @ngxson : this function may need to be improved in the future
    // handle input files
    out_files.clear();
    auto it = in_files.find("file");
    if (it != in_files.end()) {
        out_files.push_back(it->second.data);
    } else {
        throw std::invalid_argument("No input file found for transcription");
    }

    // handle input data
    std::string prompt          = json_value(inp_body, "prompt", std::string());
    std::string language        = json_value(inp_body, "language", std::string());
    std::string response_format = json_value(inp_body, "response_format", std::string("json"));
    if (response_format != "json") {
        throw std::invalid_argument("Only 'json' response_format is supported for transcription");
    }
    common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls, language, 0.0, prompt);
    if (preset.needs_audio_length) {
        mtmd_helper_media_meta meta;
        if (mtmd_helper_get_media_meta(out_files[0].data(), out_files[0].size(), mtmd_helper_init_opt_default(), &meta) != 0 ||
                meta.type != MTMD_HELPER_MEDIA_TYPE_AUDIO || meta.duration < 0) {
            throw std::invalid_argument("the input file is not a supported audio file");
        }
        preset = common_chat_get_asr_prompt(tmpls, language, meta.duration, prompt);
    }
    if (prompt.empty() || preset.has_context) {
        prompt = preset.user;
    }
    if (!language.empty() && !preset.has_language) {
        prompt += string_format(" (language: %s)", language.c_str());
    }
    prompt = preset.media_first ? get_media_marker() + prompt : prompt + get_media_marker();

    json messages = json::array();
    if (!preset.system.empty()) {
        messages.push_back({{"role", "system"}, {"content", preset.system}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json chatcmpl_body = inp_body; // copy all fields
    chatcmpl_body["messages"] = messages;

    // because input from form-data, everything is string, we need to correct the types here
    std::string stream = json_value(inp_body, "stream", std::string("false"));
    chatcmpl_body["stream"] = stream == "true";

    if (inp_body.contains("max_tokens")) {
        std::string inp = inp_body["max_tokens"].get<std::string>();
        chatcmpl_body["max_tokens"] = std::stoul(inp);
    }

    if (inp_body.contains("temperature")) {
        std::string inp = inp_body["temperature"].get<std::string>();
        chatcmpl_body["temperature"] = std::stof(inp);
    }

    return chatcmpl_body;
}

json convert_speech_to_tts(const json & inp_body) {
    json tts_body = inp_body; // copy all fields

    // a custom voice is given as {"id": "..."}
    if (inp_body.contains("voice") && inp_body.at("voice").is_object()) {
        tts_body["voice"] = json_value(inp_body.at("voice"), "id", std::string());
    }

    // the default of the OpenAI API is mp3, we have no encoder for it
    const std::string response_format = json_value(inp_body, "response_format", std::string("wav"));
    if (response_format == "pcm") {
        tts_body["response_format"] = "pcm16"; // "pcm" is float32 in /tts
    } else if (response_format == "wav") {
        tts_body["response_format"] = "wav";
    } else {
        throw std::invalid_argument("Only 'wav' and 'pcm' response_format are supported for speech");
    }

    const std::string stream_format = json_value(inp_body, "stream_format", std::string());
    if (stream_format == "audio") {
        tts_body["stream"] = true;
    } else if (!stream_format.empty()) {
        throw std::invalid_argument("Only 'audio' stream_format is supported for speech");
    }

    if (!json_value(inp_body, "instructions", std::string()).empty()) {
        SRV_WRN("%s", "'instructions' is not supported for speech and is ignored\n");
    }

    return tts_body;
}

llama_tokens server_chat_get_asr_prompt(const llama_model * model) {
    const char * src = llama_model_chat_template(model, "asr");
    if (src == nullptr) {
        return {};
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token bos = llama_vocab_bos(vocab);
    const common_chat_template tmpl(src, bos == LLAMA_TOKEN_NULL ? "" : common_token_to_piece(vocab, bos, true), "");
    jinja::context ctx(tmpl.source());
    jinja::global_from_json(ctx, json {
        {"bos_token",             tmpl.bos_token()},
        {"messages",              json::array()},
        {"add_generation_prompt", true},
    }, false);
    jinja::runtime runtime(ctx);
    const std::string prompt = jinja::runtime::gather_string_parts(runtime.execute(tmpl.prog))->as_string().str();
    return common_tokenize(vocab, prompt, false, true);
}
