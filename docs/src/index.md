# IterativeLearningControl2

[![Build Status](https://github.com/baggepinnen/IterativeLearningControl2.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/baggepinnen/IterativeLearningControl2.jl/actions/workflows/CI.yml?query=branch%3Amain)

## What is ILC?

[ILC](https://slunik.slu.se/kursfiler/TE0010/10095.1213/REG2_ILCReview.pdf) can be thought of as either
- a simple reinforcement-learning (RL) strategy, or
- a method to solve open-loop optimal control problems.

ILC is suitable in situations where a *repetitive task* is to be performed multiple times, and disturbances acting on the system are also repetitive and predictable but  may be unknown. Multiple versions of ILC exists, of which we support a few that are listed below. When ILC iterations are performed by running experiments on a physical system, ILC resembles episode-based reinforcement learning (or adaptive control), while if a model is used to simulate the experiments, we can instead think of ILC as a way to solve optimal control problems (trajectory optimization).



## Algorithms

We support the following algorithms:

```@setup ALGORITHMS
using PrettyTables, Markdown

header = ["Algorithm", "Model based", "MIMO", "Nonlinear", "Cost function", "Constraints", "Computational complexity", "Experimental complexity"]

data = [
    "HeuristicILC"        "🔶" "🟥" "🟥" "🟥" "🟥" "Low (filtering)"    "Low (1)"
    "OptimizationILC"     "🟢" "🟢" "🟢" "🟢" "🟥" "Mid (Cholesky)"     "Low (1)"
    "ConstrainedILC"      "🟢" "🟢" "🟢" "🟢" "🟢" "High (QP)"          "Low (1)"
    "GradientILC"         "🟢" "🟢" "🟢" "🔶" "🟥" "Low"                "Low (1)"
    "ModelFreeILC"        "🟥" "🟢" "🟢" "🔶" "🟥" "Low"                "High (3)"
]

io = IOBuffer()
tab = pretty_table(io, data; header, tf=tf_html_default)
tab_algs = String(take!(io)) |> HTML
```
```@example ALGORITHMS
tab_algs # hide
```

Each algorithm has an associated documentation page available from the menu on the left.

### Comments
- The 🔶 used for [`HeuristicILC`](@ref) indicates that the learning filters may be optionally chosen in a model-based way, but heuristic choices are also possible.
- Most algorithms work for time varying and/or nonlinear systems by considering linearizations around the last recorded trajectory. 
- The gradient-based algorithms, like [`GradientILC`](@ref) and [`ModelFreeILC`](@ref) can easily be modified to include a penalty on the size of the adjustment signal ``a``.
- All algorithms can be trivially modified to add the ``Q`` filter present in [`HeuristicILC`](@ref) in order to improve robustness to measurement noise and model errors.

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

## Where to apply the ILC adjustment signal?

The ILC adjustment signal ``a`` can be applied to _any input_, typically either the plant input ``u`` or the reference ``r``. The choice of where to apply the adjustment signal depends on the options afforded by the system to be controlled. 

```
                a ┌─────┐
   ┌─────────/─┬──┤ ILC │◄─┐   ◄─── Operates in batch mode
   │           /  └─────┘  │
   │           │           │
   │           │           │
r  ▼  ┌─────┐  │  ┌─────┐  │
───+─►│     │u ▼  │     │ y│
      │  C  ├──+─►│  P  ├──┼─►
    ┌►│     │     │     │  │
    │ └─────┘     └─────┘  │
    │                      │
    └──────────────────────┘
```
Several industrially occurring control systems lack the possibility to directly manipulate the plant input, in which case modifying the reference ``r`` is the only option. In this case, we chose ``G_a = G_r`` since the transfer function from ``a`` to ``y`` is the same as the transfer function from ``r`` to ``y``. A drawback of this approach is that the ILC adjustment signal ``a`` will depend on the controller, and if the controller is changed, the ILC adjustment signal will have to be recomputed. Some of the ILC algorithms make use of models of the system from ``a`` to ``y``, and the controller ``C`` typically reduces the uncertainty present in the model ``P``, _potentially_ making the model-based ILC task easier.


If ``a`` is added directly to the plant input ``u``, we have ``G_a = G_u`` and the ILC adjustment is decoupled from the feedback controller. On the other hand, applying an ILC adjustment signal ``a`` to the plant input ``u`` may require additional precautions since any safety logic present in the controller ``C`` may be circumvented.