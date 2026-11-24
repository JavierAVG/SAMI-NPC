#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sami {
namespace core {

/**
 * Configuración de hardware y parámetros base para cargar un modelo local.
 */
struct ModelLoadConfig {
    std::string model_path;          // Ruta absoluta o relativa al archivo .gguf
    int32_t context_size = 2048;     // Tamaño de ventana de contexto (tokens)
    int32_t n_threads = 4;           // Hilos de CPU (ideal: núcleos físicos disponibles)
    int32_t n_gpu_layers = 0;        // Capas descargadas a GPU (0 = solo CPU, 99 = toda la GPU)
    bool use_mmap = true;            // Mapeo de memoria para carga rápida y menor uso de RAM
};

/**
 * Parámetros de muestreo (sampling) que controlan la creatividad y determinismo del NPC.
 */
struct InferenceParams {
    int32_t max_tokens = 256;        // Límite de longitud de respuesta para controlar latencia
    float temperature = 0.7f;        // Creatividad (0.0 = determinista, >0.8 = más variado)
    float top_p = 0.9f;              // Nucleus sampling
    float repeat_penalty = 1.1f;     // Evita bucles repetitivos en modelos pequeños (<3B)
    std::vector<std::string> stop_sequences; // Secuencias de corte (ej. "<|im_end|>", "\nJugador:")
};

/**
 * Callback invocado por cada token generado.
 * Retorna `true` para continuar generando, o `false` para abortar la inferencia de inmediato.
 */
using TokenCallback = std::function<bool(const std::string& token)>;

/**
 * Interfaz pura para motores de inferencia (Model Adapter).
 * Desacopla el AI Runtime de llama.cpp, ONNX Runtime u otro backend futuro.
 */
class IModelEngine {
public:
    virtual ~IModelEngine() = default;

    // Carga el modelo en memoria/VRAM según la configuración de hardware.
    virtual bool load(const ModelLoadConfig& config) = 0;

    // Libera explícitamente el modelo y los contextos de hardware.
    virtual void unload() = 0;

    // Indica si hay un modelo listo para inferencia.
    virtual bool is_loaded() const = 0;

    /**
     * Ejecuta la inferencia de forma bloqueante en el hilo actual,
     * emitiendo tokens progresivamente a través de `on_token`.
     */
    virtual bool generate(
        const std::string& formatted_prompt,
        const InferenceParams& params,
        TokenCallback on_token
    ) = 0;

    // Utilidad para medir cuántos tokens ocupa un texto antes de enviarlo al contexto.
    virtual int32_t count_tokens(const std::string& text) const = 0;
};

} // namespace core
} // namespace sami