# GradientILC

This ILC scheme uses a model, linear or nonlinear, SISO or MIMO, to compute the gradient of a quadratic cost model
```math
\operatorname{min.}_{a} J = e^T e
```

The update to the ILC adjustment signal ``a`` is then computed as
```math
a_{k+1} = a_k + β H^T e_k
```
The parameter $\beta$ is the learning step size, ``H`` is the Jacobian of the plant output w.r.t. ``a`` and ``e`` is the tracking error.

The ILC update rule above highlights the similarity between this scheme and a simple gradient-descent algorithm to solve an optimal control problem, the only difference is whether the tracking error ``e`` comes from a simulation (optimal control) or from an experiment on a physical system (ILC). The cost model can also be trivially modified to include a penalty on the size of the adjustment signal ``a``.

A model-free version of this algorithm exists, see [ModelFreeILC](@ref).


# Example

This example mirrors that of [HeuristicILC](@ref), we create the system model and feedback controller here without any explanation, and refer to the [HeuristicILC](@ref) example for those details
```@example MODELFREE_ILC
using IterativeLearningControl2, ControlSystemsBase, Plots

function double_mass_model(; 
                Jm = 1,   # motor inertia
                Jl = 1,   # load inertia
                k  = 100, # stiffness
                c0 = 1,   # motor damping
                c1 = 1,   # transmission damping
                c2 = 1,   # load damping
)

    A = [
        0.0 1 0 0
        -k/Jm -(c1 + c0)/Jm k/Jm c1/Jm
        0 0 0 1
        k/Jl c1/Jl -k/Jl -(c1 + c2)/Jl
    ]
    B = [0, 1/Jm, 0, 0]
    C = [1 0 0 0]
    ss(A, B, C, 0)
end

# Continuous
P    = double_mass_model(Jl = 1)
Pact = double_mass_model(Jl = 1.5) # 50% more load than modeled
C    = pid(10, 1, 1, form = :series) * tf(1, [0.02, 1])

Ts = 0.02 # Sample time
Gr = c2d(feedback(P*C), Ts)       |> tf
Gu = c2d(feedback(P, C), Ts)
Gract = c2d(feedback(Pact*C), Ts)
Guact = c2d(feedback(Pact, C), Ts)

T = 3pi    # Duration
t = 0:Ts:T # Time vector
function funnysin(t)
    x = sin(t)
    s,a = sign(x), abs(x)
    y = s*((a + 0.01)^0.2 - 0.01^0.2)
    t > 2π ? sign(y) : y
end
r = funnysin.(t)' |> Array # Reference signal
```

Next, we define the [`ILCProblem`](@ref) and create the learning algorithm object [`OptimizationILC`](@ref)
```@example MODELFREE_ILC
prob = ILCProblem(; r, Gr, Gu)
actual = ILCProblem(; r, Gr=Gract, Gu=Guact)

alg = GradientILC(500)
sol = ilc(prob, alg; iters=5)
plot(sol)
```
The result looks good, how about when run on the "actual" dynamics with 50% larger load inertia?
    
```@example MODELFREE_ILC
sol = ilc(prob, alg; actual, iters=5)
plot(sol)
```
Still pretty good.



## Docstring
    
```@docs
GradientILC
```