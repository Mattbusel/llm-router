# llm-router

Route prompts to the right model by complexity. Single header, no deps.

## Quickstart

`cpp
#define LLM_ROUTER_IMPLEMENTATION
#include "llm_router.hpp"

llm::RouterConfig cfg;
cfg.strategy = llm::RoutingStrategy::Balanced;
cfg.models = {{"gpt-4o-mini", 0.15, 0.5, 0.7, 40}, {"gpt-4o", 5.0, 1.0, 0.9, 100}};
llm::Router router(cfg);
auto d = router.route("Explain quantum physics in detail");
std::cout << d.model_name << "\n";
`

## Strategies: CostOptimized, LatencyOptimized, QualityOptimized, Balanced, Budget

## Build

`ash
cmake -B build
cmake --build build
`

## Examples

| File | Description |
|------|-------------|
| basic_router.cpp | Balanced routing |
| budget_routing.cpp | Hard cost cap |
| learning_router.cpp | EMA adaptation |
| stats.cpp | Complexity scores |

## See Also

| Repo | What it does |
|------|-------------|
| [llm-stream](https://github.com/Mattbusel/llm-stream) | Stream OpenAI and Anthropic responses via SSE |
| [llm-cache](https://github.com/Mattbusel/llm-cache) | LRU response cache |
| [llm-cost](https://github.com/Mattbusel/llm-cost) | Token counting and cost estimation |
| [llm-retry](https://github.com/Mattbusel/llm-retry) | Retry and circuit breaker |
| [llm-format](https://github.com/Mattbusel/llm-format) | Structured output / JSON schema |
| [llm-embed](https://github.com/Mattbusel/llm-embed) | Embeddings and vector search |
| [llm-pool](https://github.com/Mattbusel/llm-pool) | Concurrent request pool |
| [llm-log](https://github.com/Mattbusel/llm-log) | Structured JSONL logging |
| [llm-template](https://github.com/Mattbusel/llm-template) | Prompt templating |
| [llm-agent](https://github.com/Mattbusel/llm-agent) | Tool-calling agent loop |
| [llm-rag](https://github.com/Mattbusel/llm-rag) | RAG pipeline |
| [llm-eval](https://github.com/Mattbusel/llm-eval) | Evaluation and consistency scoring |
| [llm-chat](https://github.com/Mattbusel/llm-chat) | Conversation memory manager |
| [llm-vision](https://github.com/Mattbusel/llm-vision) | Multimodal image+text |
| [llm-mock](https://github.com/Mattbusel/llm-mock) | Mock LLM for testing |
| [llm-router](https://github.com/Mattbusel/llm-router) | Model routing by complexity |
| [llm-guard](https://github.com/Mattbusel/llm-guard) | PII detection and injection guard |
| [llm-compress](https://github.com/Mattbusel/llm-compress) | Context compression |
| [llm-batch](https://github.com/Mattbusel/llm-batch) | Batch processing and checkpointing |

## License

MIT -- Copyright (c) 2026 Mattbusel. See LICENSE.
