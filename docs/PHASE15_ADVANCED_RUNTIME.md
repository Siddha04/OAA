# OAA Phase 15 - Advanced Runtime and Agent Adapter Contracts

## What this phase adds

- BackendCapabilities is an explicit, validated inventory of modality support, checkpoint status, true streaming, structured tool-call support, quantization and known context length.
- RuntimeEngineBackend describes the current native tiny transformer honestly: seeded in-memory float32 weights, text only, no true streaming or structured tool calls.
- GenerationService validates prompt size, token-generation budget, backend type and output size; GenerationResult reports elapsed time and characters-per-second without embedding raw prompt/output text in metrics.
- AgentPlanProvider allows an explicitly injected planner to propose structured plans using registered tool schemas. The plan is validated before use and is never automatically executed. Existing tool policy and approval checks remain authoritative.
- genai runtime-info reports the real capabilities of the loaded current backend and the known limitations.

## Usage

    genai runtime-info --model ./tiny.manifest

Python generation service:

    from oaa import Engine, GenerationConfig, GenerationService, RuntimeEngineBackend

    engine = Engine()
    engine.load_model("./tiny.manifest")
    service = GenerationService(RuntimeEngineBackend(engine), max_output_tokens=128)
    result = service.generate("Hello", GenerationConfig(max_tokens=16))
    print(result.text)
    print(result.metrics())

A future provider can implement propose_plan(goal, tools) and return a structured plan. The current agent validates the result against registered schemas; policy checks and per-step approval are still required to execute it. No provider is chosen automatically.

## Limitations

This phase improves extension safety and diagnostics but does not add pretrained checkpoint inference, true incremental streaming, model quantization, CUDA transformer kernels, an optimizer, a local LLM planner, or multimodal model weights. The report from runtime-info should be treated as the backend's current capability contract rather than a claim of production readiness.
