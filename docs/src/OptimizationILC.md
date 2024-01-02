# OptimizationILC

This ILC algorithm is derived by considering the optimization problem
```math
\operatorname{min}_{a_{k+1}} J_{k+1} = e_{k+1}^T e_{k+1} + ρ a_{k+1}^T a_{k+1}
```
subject to
```math
||a_{k+1}-a_{k}||_2^2 < \delta
```

Internally, a system model on Hankel-operator form is used, that is, the linear system is represented as a matrix operator ``T_a`` consisting of shifted impulse responses such that the relation between the ILC adjustment signal ``a`` and the output ``y`` is given by
```math
y = T_a a
```


After some derivations, available in Norrlöf's thesis, a simple algorithm is obtained that relies on only matrix factorizations:
```julia
Q = ((ρ+λ)*I + Ta'Ta)\(λ*I + Ta'Ta)
L = (λ*I + Ta'Ta)\Ta'
a[k+1] = (Q*(a[k]' + L*e'))'
```
where the tuning variable ``λ`` is used to control the learning rate, and ``ρ`` is used to control the trade-off between the control error ``e`` and control effort ``a``.


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

Next, we define the [`ILCProblem`](@ref) and create the learning algorithm object [`OptimizationILC`](@ref)
```@example OPTIMIZATION_ILC
prob = ILCProblem(; r, Gr, Gu)
alg = OptimizationILC(; ρ=0.00001, λ=0.0001)
sol = ilc(prob, alg)
plot(sol)
```
The result looks good when run on the model, but how does it looks if we run it on the "actual" dynamics with 50% larger load inertia?

```@example OPTIMIZATION_ILC
actual = ILCProblem(; r, Gr=Gract, Gu=Guact)
sol = ilc(prob, alg; actual)
plot(sol)
```
Still quite good. The resulting ILC feedforward signal ``a`` in this example is not immediately comparable to that from the [HeuristicILC](@ref) example. Here, we let the ILC signal enter directly at the plant input, while in the [HeuristicILC](@ref) example, the ILC signal was added to the reference signal.


## Docstring
    
```@docs
OptimizationILC
```