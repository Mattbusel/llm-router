# llm-router

[![CI](https://github.com/Mattbusel/llm-router/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-router/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Send each prompt to the right model: cheap for easy ones, strong for hard ones.** One header, `llm_router.hpp`. Needs nothing beyond the C++17 standard library.

Most traffic does not need your most expensive model. llm-router scores each prompt's complexity with fast local heuristics and picks a model from your list according to a strategy: cheapest capable, fastest, best quality, balanced, or capped by budget. It can also learn from outcome feedback. It makes no network calls; pair it with llm-stream or your own client to send the request.

## Features

- `score_complexity()`: 0 to 100 from length, question count, technical keywords and other cues
- Strategies: `CostOptimized`, `LatencyOptimized`, `QualityOptimized`, `Balanced`, `Budget`
- Per-model profile: relative cost, latency score, quality score, maximum complexity it handles well
- `RouteDecision` explains the choice (model, complexity score, reason)
- Optional learning: `record_outcome()` adjusts each model's quality with an exponential moving average
- Offline, standard library only

## Quick start

Copy the header into your project:

```bash
curl -fsSLO https://raw.githubusercontent.com/Mattbusel/llm-router/main/include/llm_router.hpp
```

Define `LLM_ROUTER_IMPLEMENTATION` in exactly one `.cpp` file before including it; every other file just includes the header. Save this as `main.cpp` next to the header:

```cpp
#define LLM_ROUTER_IMPLEMENTATION
#include "llm_router.hpp"
#include <iostream>

int main() {
    llm::RouterConfig cfg;
    cfg.strategy = llm::RoutingStrategy::Balanced;
    cfg.models = {
        // name,         cost/1k, latency, quality, max complexity
        {"gpt-4o-mini",  0.15,    0.5,     0.7,     40},
        {"gpt-4o",       5.0,     1.0,     0.9,     80},
        {"big-model",    15.0,    1.5,     1.0,     100},
    };
    cfg.learn_from_outcomes = true;

    llm::Router router(cfg);
    for (std::string p : {"Hi",
                          "Implement and optimize a lock-free queue in C++. "
                          "Explain the memory ordering and prove it is correct?"}) {
        auto d = router.route(p);
        std::cout << d.model_name << "  complexity=" << d.complexity_score
                  << "  (" << d.reason << ")\n";
    }

    router.record_outcome("gpt-4o-mini", 0.4);   // feed back observed quality, 0..1
    std::cout << router.effective_quality("gpt-4o-mini") << "\n";
}
```

```bash
g++ -std=c++17 -O2 main.cpp -o demo
```

## API at a glance

| Call | Purpose |
|---|---|
| `score_complexity(prompt)` | Heuristic complexity, 0 to 100 |
| `Router(RouterConfig)` | Strategy, model list, budget tier, learning settings |
| `route(prompt)` | `RouteDecision{model_name, complexity_score, reason}` |
| `record_outcome(model, quality)` | Feed back a 0 to 1 quality score |
| `effective_quality(model)` | Quality after learning |

## Notes and limitations

- Complexity scoring is keyword and length based, not a model. It is fast and predictable, but tune your model thresholds against your own traffic.
- The header defines `NOMINMAX` so it is safe to include alongside `<windows.h>`.

## Build the examples

The repo builds `examples/basic_router.cpp`, `examples/budget_routing.cpp`, `examples/learning_router.cpp`, `examples/stats.cpp` with CMake:

```bash
cmake -B build
cmake --build build
```

## Part of llm-cpp

llm-router is one of 26 single-header C++ libraries in [llm-cpp](https://github.com/Mattbusel/llm-cpp), a toolkit for building LLM features into native code. Each library stands alone; combine them by giving each `*_IMPLEMENTATION` define its own `.cpp` file. See the [llm-cpp README](https://github.com/Mattbusel/llm-cpp#using-several-together) for the full list and examples of using several together.

## License

MIT. See [LICENSE](LICENSE).
