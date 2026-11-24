#pragma once

#include "IModelEngine.h"

// Forward declarations para aislar completamente llama.h de los consumidores
struct llama_model;
struct llama_context;

namespace sami {
namespace core {

class LlamaCppAdapter : public IModelEngine {
public:
    LlamaCppAdapter();
    ~LlamaCppAdapter() override;

    // Evitar copias para proteger los punteros nativos de C
    LlamaCppAdapter(const LlamaCppAdapter&) = delete;
    LlamaCppAdapter& operator=(const LlamaCppAdapter&) = delete;

    bool load(const ModelLoadConfig& config) override;
    void unload() override;
    bool is_loaded() const override;

    bool generate(
        const std::string& formatted_prompt,
        const InferenceParams& params,
        TokenCallback on_token
    ) override;

    int32_t count_tokens(const std::string& text) const override;

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    ModelLoadConfig current_config_;

    // Convierte un ID de token numérico a su representación cruda de bytes
    std::string token_to_piece(int32_t token_id) const;
};

} // namespace core
} // namespace sami