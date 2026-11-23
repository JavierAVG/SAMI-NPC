# SAMI-NPC
SAMI-NPC (Small Agnostic Model Integration for Non Playable Characters) es un SDK modular, distribuido como un plugin para Godot 4 dirigido a la integración de Small Language Models en hardware de gama media-baja para el desarrollo de interacciones conversacionales con NPCs, abarcando herramientas de memoria, contexto, interacción y personalidad.

Estructura de repositorio
```
/SAMI-NPC
├── /core                           # Capas Agnósticas (Compila a una DLL/SO base)
│   ├── /model_adapter
│   │   ├── /third_party            # Submódulo de llama.cpp u ONNX Runtime
│   │   ├── /include
│   │   │   └── IModelEngine.h      # Interfaz pura para poder cambiar el motor futuro
│   │   └── /src
│   │       └── LlamaCppAdapter.cpp # Implementación específica
│   ├── /ai_runtime
│   │   ├── /include
│   │   │   ├── NPCSession.h        # Lógica conversacional, memoria, RAG (si aplica)
│   │   │   └── ContextManager.h    # Gestión de prompts del sistema y estado
│   │   └── /src
│   │       └── NPCSession.cpp
│   └── CMakeLists.txt              # Sistema de compilación del core independiente
│
├── /engine_adapters                # Puentes hacia los motores (Plugins)
│   └── /godot_gdextension
│       ├── /src
│       │   ├── GD_NPCSession.h     # Wrapper de GDExtension para NPCSession.h
│       │   └── GD_NPCSession.cpp 
│       └── SConstruct              # Build script de Godot
│
└── /slm-npc-test-project           # Proyecto de prueba y distribución final
    ├── /addons
    │   └── /slm_npc                # El plugin final que el desarrollador instala
    │       ├── /bin                # Binarios compilados (.dll, .so, .dylib)
    │       ├── /models             # Modelos .gguf (excluidos del control de versiones)
    │       ├── slm_npc.gdextension # Manifiesto de GDExtension
    │       └── NPC_Node.gd         # Scripts auxiliares para el desarrollador
    ├── export_presets.cfg          # Configuración crítica para exportar los modelos
    └── project.godot
```