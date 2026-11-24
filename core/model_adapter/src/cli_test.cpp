#include "LlamaCppAdapter.h"
#include <chrono>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: ./cli_test <ruta_al_modelo.gguf> [n_gpu_layers]\n";
        return 1;
    }

    sami::core::ModelLoadConfig load_cfg;
    load_cfg.model_path = argv[1];
    load_cfg.context_size = 1024;
    load_cfg.n_threads = 4;
    load_cfg.n_gpu_layers = (argc >= 3) ? std::stoi(argv[2]) : 0;

    std::unique_ptr<sami::core::IModelEngine> engine = std::make_unique<sami::core::LlamaCppAdapter>();

    std::cout << "Cargando modelo desde: " << load_cfg.model_path << "\n";
    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!engine->load(load_cfg)) {
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << "Modelo cargado en " << load_ms << " ms.\n\n";

    // Prompt formateado estilo ChatML (compatible con Qwen2.5, Phi-3, SmolLM2, etc.)
    std::string prompt =
        "<|im_start|>system\n"
        "Eres un herrero gruñón de una aldea medieval. Responde en español en máximo 2 oraciones cortas.<|im_end|>\n"
        "<|im_start|>user\n"
        "¡Hola maestro! ¿Podrías reparar mi espada oxidada?<|im_end|>\n"
        "<|im_start|>assistant\n";

    sami::core::InferenceParams inf_params;
    inf_params.max_tokens = 120;
    inf_params.temperature = 0.7f;
    inf_params.stop_sequences = {"<|im_end|>"};

    bool first_token = true;
    int chunks_received = 0;
    auto t_gen_start = std::chrono::high_resolution_clock::now();
    double ttft_ms = 0.0;

    std::cout << "NPC: ";
    bool ok = engine->generate(prompt, inf_params, [&](const std::string& token) {
        if (first_token) {
            auto t_first = std::chrono::high_resolution_clock::now();
            ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_gen_start).count();
            first_token = false;
        }
        std::cout << token << std::flush;
        chunks_received++;
        return true;
    });

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
    double tps = (chunks_received > 0 && total_ms > 0) ? (chunks_received / (total_ms / 1000.0)) : 0.0;

    std::cout << "\n\n--- Métricas de Rendimiento ---\n"
              << "Estado: " << (ok ? "OK" : "Error") << "\n"
              << "TTFT (Latencia 1er token): " << ttft_ms << " ms\n"
              << "Tiempo total de respuesta: " << total_ms << " ms\n"
              << "Velocidad aproximada: " << tps << " tokens/s\n";

    return ok ? 0 : 1;
}