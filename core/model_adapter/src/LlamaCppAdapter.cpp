#include "LlamaCppAdapter.h"
#include "llama.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace sami {
namespace core {

namespace {

/**
 * Calcula cuántos bytes iniciales de `str` forman secuencias UTF-8 completas.
 * Evita emitir bytes sueltos de caracteres multibyte (tildes, ñ, ¿, ¡) a mitad de un token.
 */
size_t valid_utf8_prefix_length(const std::string& str) {
    size_t i = 0;
    const size_t len = str.size();

    while (i < len) {
        const unsigned char c = static_cast<unsigned char>(str[i]);
        size_t char_len = 0;

        if (c < 0x80) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        } else {
            i += 1;
            continue;
        }

        if (i + char_len > len) {
            break;
        }
        i += char_len;
    }

    return i;
}

} // namespace

LlamaCppAdapter::LlamaCppAdapter() {
    ggml_backend_load_all();
}

LlamaCppAdapter::~LlamaCppAdapter() {
    unload();
}

bool LlamaCppAdapter::load(const ModelLoadConfig& config) {
    if (is_loaded()) {
        unload();
    }

    current_config_ = config;

    // 1. Configuración y carga de pesos del modelo (.gguf)
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_params.load_mode = config.use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;

    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model_) {
        std::cerr << "[LlamaCppAdapter] Error: No se pudo cargar el modelo en: "
                  << config.model_path << "\n";
        return false;
    }

    // 2. Configuración e instanciación del contexto (KV-Cache y hilos)
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(config.context_size);
    ctx_params.n_batch = static_cast<uint32_t>(std::min(config.context_size, 512));
    ctx_params.n_threads = config.n_threads;
    ctx_params.n_threads_batch = config.n_threads;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        std::cerr << "[LlamaCppAdapter] Error: No se pudo crear el contexto de inferencia.\n";
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    return true;
}

void LlamaCppAdapter::unload() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

bool LlamaCppAdapter::is_loaded() const {
    return model_ != nullptr && ctx_ != nullptr;
}

int32_t LlamaCppAdapter::count_tokens(const std::string& text) const {
    if (!model_ || text.empty()) {
        return 0;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model_);
    const int32_t n_tokens = -llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        true,
        true
    );

    return std::max(0, n_tokens);
}

std::string LlamaCppAdapter::token_to_piece(int32_t token_id) const {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    char buf[256];
    const int32_t n = llama_token_to_piece(vocab, token_id, buf, sizeof(buf), 0, true);

    if (n < 0) {
        std::vector<char> large_buf(static_cast<size_t>(-n));
        const int32_t n2 = llama_token_to_piece(
            vocab, token_id, large_buf.data(), static_cast<int32_t>(large_buf.size()), 0, true
        );
        return (n2 > 0) ? std::string(large_buf.data(), static_cast<size_t>(n2)) : "";
    }

    return std::string(buf, static_cast<size_t>(n));
}

bool LlamaCppAdapter::generate(
    const std::string& formatted_prompt,
    const InferenceParams& params,
    TokenCallback on_token
) {
    if (!is_loaded()) {
        std::cerr << "[LlamaCppAdapter] Error: Intento de generación sin modelo cargado.\n";
        return false;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model_);

    // 1. Tokenizar el prompt completo
    const int32_t n_prompt_tokens = -llama_tokenize(
        vocab,
        formatted_prompt.c_str(),
        static_cast<int32_t>(formatted_prompt.size()),
        nullptr,
        0,
        true,
        true
    );

    if (n_prompt_tokens <= 0) {
        std::cerr << "[LlamaCppAdapter] Error: Prompt vacío o fallo en tokenización.\n";
        return false;
    }

    const uint32_t n_ctx = llama_n_ctx(ctx_);
    if (static_cast<uint32_t>(n_prompt_tokens) >= n_ctx) {
        std::cerr << "[LlamaCppAdapter] Error: El prompt (" << n_prompt_tokens
                  << " tokens) excede el tamaño de contexto (" << n_ctx << ").\n";
        return false;
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(n_prompt_tokens));
    llama_tokenize(
        vocab,
        formatted_prompt.c_str(),
        static_cast<int32_t>(formatted_prompt.size()),
        prompt_tokens.data(),
        n_prompt_tokens,
        true,
        true
    );

    // 2. Limpiar memoria KV de turnos anteriores
    llama_memory_clear(llama_get_memory(ctx_), true);

    // 3. Configurar cadena de muestreo (Samplers)
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_sampler_chain_add(
        smpl,
        llama_sampler_init_penalties(n_vocab, 64, params.repeat_penalty, 0.0f, 0.0f)
    );

    if (params.temperature <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    // 4. Evaluar el prompt inicial en lotes (Prefill)
    const int32_t n_batch = static_cast<int32_t>(llama_n_batch(ctx_));
    for (int32_t i = 0; i < n_prompt_tokens; i += n_batch) {
        const int32_t chunk_size = std::min(n_batch, n_prompt_tokens - i);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, chunk_size);
        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "[LlamaCppAdapter] Error en llama_decode durante el prefill.\n";
            llama_sampler_free(smpl);
            return false;
        }
    }

    size_t max_stop_len = 0;
    for (const auto& seq : params.stop_sequences) {
        max_stop_len = std::max(max_stop_len, seq.size());
    }

    std::string utf8_buffer;
    std::string emission_buffer;
    bool aborted_or_stopped = false;

    // 5. Bucle autoregresivo de generación token a token
    int32_t n_cur = n_prompt_tokens;
    const int32_t max_total_tokens = std::min(
        static_cast<int32_t>(n_ctx),
        n_prompt_tokens + params.max_tokens
    );

    while (n_cur < max_total_tokens) {
        const llama_token new_token_id = llama_sampler_sample(smpl, ctx_, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        utf8_buffer += token_to_piece(new_token_id);
        const size_t valid_len = valid_utf8_prefix_length(utf8_buffer);
        if (valid_len > 0) {
            emission_buffer.append(utf8_buffer, 0, valid_len);
            utf8_buffer.erase(0, valid_len);
        }

        size_t earliest_stop_pos = std::string::npos;
        for (const auto& stop_seq : params.stop_sequences) {
            if (stop_seq.empty()) continue;
            const size_t pos = emission_buffer.find(stop_seq);
            if (pos != std::string::npos) {
                earliest_stop_pos = std::min(earliest_stop_pos, pos);
            }
        }

        if (earliest_stop_pos != std::string::npos) {
            if (earliest_stop_pos > 0 && on_token) {
                on_token(emission_buffer.substr(0, earliest_stop_pos));
            }
            emission_buffer.clear();
            aborted_or_stopped = true;
            break;
        }

        if (emission_buffer.size() > max_stop_len) {
            const size_t safe_emit_len = emission_buffer.size() - max_stop_len;
            const std::string chunk = emission_buffer.substr(0, safe_emit_len);
            emission_buffer.erase(0, safe_emit_len);

            if (on_token && !on_token(chunk)) {
                aborted_or_stopped = true;
                break;
            }
        }

        llama_token next_token = new_token_id;
        llama_batch next_batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx_, next_batch) != 0) {
            std::cerr << "[LlamaCppAdapter] Error en llama_decode durante generación.\n";
            llama_sampler_free(smpl);
            return false;
        }

        n_cur++;
    }

    if (!aborted_or_stopped && !emission_buffer.empty() && on_token) {
        on_token(emission_buffer);
    }

    llama_sampler_free(smpl);
    return true;
}

} // namespace core
} // namespace sami