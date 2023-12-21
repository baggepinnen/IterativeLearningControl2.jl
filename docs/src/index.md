# IterativeLearningControl

[![Build Status](https://github.com/baggepinnen/IterativeLearningControl.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/IterativeLearningControl.jl/actions/workflows/CI.yml?query=branch%3Amain)

## What is ILC?

[ILC](https://slunik.slu.se/kursfiler/TE0010/10095.1213/REG2_ILCReview.pdf) can be thought of as either
- a simple reinforcement-learning (RL) strategy, or
- a method to solve open-loop optimal control problems.

ILC is suitable in situations where a *repetitive task* is to be performed multiple times, and disturbances acting on the system are also repetitive and predictable but  may be unknown. Multiple versions of ILC exists, of which we support a few that are listed below. When ILC iterations are performed by running experiments on a physical system, ILC resembles episode-based reinforcement learning (or adaptive control), while if a model is used to simulate the experiments, we can instead think of ILC as a way to solve optimal control problems (trajectory optimization).



## Algorithms

We support the following algorithms:

```@setup ALGORITHMS
using PrettyTables, Markdown

header = ["Algorithm", "Model based", "MIMO", "Cost function", "Constraints", "Computational complexity"]

data = [
    "HeuristicILC"        "🔶" "🟥" "🟥" "🟥" "Low 🚀 (filtering)"
    "OptimizationILC"     "🟢" "🟥" "🟢" "🟥" "Medium 🤔 (matrix factorization)"
    "ConstrainedILC"      "🟢" "🟢" "🟢" "🟢" "High 🏋️ (quadratic program)"
]

io = IOBuffer()
tab = pretty_table(io, data; header, tf=tf_html_default)
tab_algs = String(take!(io)) |> HTML
```
```@example ALGORITHMS
tab_algs # hide
```

Each algorithm has an associated documentation page available from the menu on the left. The 🔶 used for [`HeuristicILC`](@ref) indicates that the learning filters may be optionally chosen in a model-based way, but heuristic choices are also possible.

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


