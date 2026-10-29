#pragma once

int llama_build_number(void);

const char * llama_commit(void);
const char * llama_compiler(void);

const char * llama_build_target(void);
const char * llama_build_info(void);

const char * alphallama_version(void);        // semantic version, e.g. "2.0.0"
const char * alphallama_version_name(void);   // release name, e.g. "AlphaLlama v2 Flash"

void llama_print_build_info(void);
