# IterativeLearningControl

[![Build Status](https://github.com/baggepinnen/IterativeLearningControl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/IterativeLearningControl.jl/actions/workflows/CI.yml?query=branch%3Amain)

## What is ILC?

ILC can be thought of as a simple reinforcement-learning strategy that is suitable in situations where a *repetitive task* is to be performed multiple times, and disturbances acting on the system are also repetitive and predictable but unknown. Multiple versions of ILC exists, of which we support a few that are listed below.


## Algorithms

We support the following algorithms:

```@setup ALGORITHMS
using PrettyTables, Markdown

header = ["Algorithm", "Model based", "MIMO", "Cost function", "Constraints", "Computational complexity"]

data = [
    "HeuristicILC"        "🔶" "🟥" "🟥" "🟥" "Low (filtering)"
    "OptimizationILC"     "🟢" "🟥" "🟢" "🟥" "Medium (matrix factorization)"
    "ConstrainedILC"      "🟢" "🟢" "🟢" "🟢" "High (quadratic program)"
]

io = IOBuffer()
tab = pretty_table(io, data; header, tf=tf_html_default)
tab_algs = String(take!(io)) |> HTML
```
```@example ALGORITHMS
tab_algs # hide
```


## Terminology
In this documentation, we will refer to the following signals and (discrete-time) transfer functions

- ``y`` is an output to be controlled
- ``r`` is a reference signal that we want ``y`` to track.
- ``u`` is the control signal
- ``a`` is the ILC adjustment signal which may be added to either the reference or directly to the plant input.
- ``P(z)`` is the plant, i.e., the system to control
- ``C(z)`` is a feedback controller
- ``G_r(z)`` is the closed-loop transfer function from ``r`` to ``y``: ``PC / (1 + PC)``
- ``G_u(z)`` is the closed-loop transfer function from ``u`` to ``y``: ``P / (1 + PC)``


