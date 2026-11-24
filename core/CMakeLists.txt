cmake_minimum_required(VERSION 3.16)
project(sami_core VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Requerido para enlazar luego en GDExtension (.so)

# Desactivar componentes innecesarios de llama.cpp y compilar estáticamente
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Opcional: Descomentar para habilitar aceleración por GPU genérica vía Vulkan
# set(GGML_VULKAN ON CACHE BOOL "" FORCE)

add_subdirectory(model_adapter/third_party/llama.cpp)

# --- Librería estática: Model Adapter ---
add_library(sami_model_adapter STATIC
    model_adapter/src/LlamaCppAdapter.cpp
)

target_include_directories(sami_model_adapter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/model_adapter/include
)

target_link_libraries(sami_model_adapter
    PRIVATE
        llama
)

# --- Ejecutable de validación en consola (sin Godot) ---
add_executable(cli_test
    model_adapter/src/cli_test.cpp
)

target_link_libraries(cli_test
    PRIVATE
        sami_model_adapter
)