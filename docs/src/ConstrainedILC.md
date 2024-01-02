# ConstrainedILC

This algorithm uses a quadratic program to solve the ILC problem subject to constraints on the adjustment signal ``a`` and the plant output ``y``. 

The algorithm comes from the paper [On Robustness in Optimization-Based Constrained Iterative Learning Control](https://arxiv.org/abs/2203.05291) and considers an LQR style optimization problem
```math
\operatorname{min}_{a_{k+1}} J_{k+1} = e_{k+1}^T Q e_{k+1} + a_{k+1}^T R a_{k+1}
```
subject to constraints on ``a`` and ``y``.

The optimization problem is internally modeled using [JuMP](https://jump.dev/JuMP.jl/stable/) and will, if the user-provided constraints are linear, end up as a quadratic program. However, the user may supply arbitrary constraints supported by JuMP, in which case a supporting optimization solver must be used. Below, we use the QP solver OSQP.


!!! note
    This algorithm is only available if the user has manually installed and loaded the packages JuMP and BlockArrays. The user must additionally install a solver compatible with the modeled optimization problem. If only linear constraints are used, we recommend OSQP.

## Constraints

The constraints are added by providing two functions to the constructor of [`ConstrainedILC`](@ref). Each of these functions take a JuMP model as well as a JuMP optimization variable as input. The user then adds the desired constraints that applies to the variable in the function. For example, to add constraints to the adjustment signal ``-25 ≤ a ≤ 25``, we create the function
```julia
A = function (model, a)
    lower,upper = -25ones(Gu.nu), 25ones(Gu.nu)
    JuMP.@constraint(model, [i=1:size(v, 2)], lower .<= a[:, i] .<= upper)
end
```

Similarly, we may add constraints on the outputs ``y`` by providing a similar function ``Y``.

To constrain the total plant input, i.e., the sum of the contributions from ILC feedforward and feedback control, we create a _constrained system_ that as the signals that we wish to constrain as outputs, and we can thus add these constraints as output constraints. If such a constrained output system is not supplied, it is assumed that the default plant output is constrained.

# Example

This example mirrors that of [HeuristicILC](@ref), we create the system model and feedback controller here without any explanation, and refer to the [HeuristicILC](@ref) example for those details
```@example OPTIMIZATION_ILC
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

Next, we define the [`ILCProblem`](@ref) and create the learning algorithm object [`ConstrainedILC`](@ref).

Here, we constrain the ILC input ``-25 ≤ a ≤ 25`` and the plant output ``-1.1 ≤ y ≤ 1.1``. We also use the QP solver OSQP. The weight matrices that penalize the control error, ``Q``, and the control effort, ``R``, are also supplied. ``α`` is a learning-rate parameter and this must be smaller than 2. The default value is 0.5.

We finally run the ILC iterations using the function [`ilc`](@ref) and plot the result.
```@example OPTIMIZATION_ILC
using JuMP, BlockArrays, OSQP, LinearAlgebra

prob = ILCProblem(; r, Gr, Gu)

Q = 1000I(Gr.ny)
R = 0.001I(Gu.nu)

A = function (model, v)
    lower,upper = -25ones(Gu.nu), 25ones(Gu.nu)
    JuMP.@constraint(model, [i=1:size(v, 2)], lower .<= v[:, i] .<= upper)
end

Y = function (model, yh)
    lower,upper = -1.1ones(Gr.ny), 1.1ones(Gr.ny)
    JuMP.@constraint(model, [i=1:size(yh, 2)], lower .<= yh[:, i] .<= upper)
end

alg = ConstrainedILC(; Q, R, A, Y, opt=OSQP.Optimizer, verbose=true, α=1)
sol = ilc(prob, alg)
plot(sol); hline!([1.1], l=(:red, :dash), sp=1, lab="Constraint")
```
We see that the output constraint is violated in the first iteration. This is expected since the optimizer hasn't yet been run while this experiment was performed. Subsequent iterations respect the constraint. 

The result looks good when run on the model, but how does it looks if we run it on the "actual" dynamics with 50% larger load inertia?

```@example OPTIMIZATION_ILC
actual = ILCProblem(; r, Gr=Gract, Gu=Guact)
sol = ilc(prob, alg; actual)
plot(sol); hline!([1.1], l=(:red, :dash), sp=1, lab="Constraint")
```
Still quite good, but we do not quite satisfy the output constraint until after 4 iterations. The paper from which the algorithm is taken contains additional considerations required for robust constraint satisfaction under bounded disturbances and model uncertainty. These are not implemented here, but we encourage the interested reader to read the paper and consider the package [LazySets.jl](https://github.com/JuliaReach/LazySets.jl) for the required computations of constraint sets, in particular, the [Minkowski difference](https://juliareach.github.io/LazySets.jl/dev/lib/binary_functions/#Minkowski-difference).

## Constraining the total control input
Above, we placed a constraint on the ILC adjustment signal ``a``, but no constraint on the total control signal including the contribution of the feedback controller. Below, we create an augmented system that includes this signal as output, and then constrain this as an output constraint. The transfer function from reference to control signal is ``C/(1+PC)``, while the transfer function from a feedforward signal added directly to the plant input to the total plant input is given by the (input) complementary sensitivity function ``CP / (1 + CP)``. We constrain the total control signal to be ``-500 ≤ u ≤ 500``, and thus modify the `Y` function from above to include this constraint:
```@example OPTIMIZATION_ILC
Gr_constraints = [Gr; c2d(feedback(C, P), Ts)] # Add output signal corresponding to the total control signal
Gu_constraints = [Gu; c2d(feedback(C*P), Ts)]

Y = function (model, yh)
    upper = [1.1, 500] # [plant output, total control signal]
    lower = -upper
    JuMP.@constraint(model, [i=1:size(yh, 2)], lower .<= yh[:, i] .<= upper)
end

alg = ConstrainedILC(; Gr_constraints, Gu_constraints, Q, R, A, Y, opt=OSQP.Optimizer, verbose=true, α=1)
sol = ilc(prob, alg; actual)
plot(sol)
```
This time, we do not achieve as small control error as before, but this is expected since the ILC algorithm now has additional constrained to respect. 

To plot the total control signal, we may simulate the augmented system. The input to this system is the reference signal ``r`` as well as the last adjustment signal ``a`` from the ILC algorithm (obtained from `sol.A[end]`)
```@example OPTIMIZATION_ILC
constrained_res = lsim([Gr_constraints Gu_constraints], [r; sol.A[end]])
plot(constrained_res, title=["Plant output" "Total control signal"]); hline!([1.1 500], l=(:red, :dash), lab="Constraint")
```
We see that the control signal spikes at the sharp step in the reference, but it stays below the constraint 500. Nice.

## Docstring
    
```@docs
ConstrainedILC
```