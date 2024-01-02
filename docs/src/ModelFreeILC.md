# ModelFreeILC

This ILC scheme is completely model free and works for nonlinear and time varying MIMO systems. The algorithm is described in the paper "Model-free Gradient Iterative Learning Control for Non-linear Systems" by Huo and friends.

The algorithm works by performing 3 experiments per ILC iteration
1. One rollout with the current ILC signal, $G(a)$
2. One rollout with the mean of the current ILC signal, $G(\hat a)$
3. One rollout with the mean perturbed by a multiple of the time-reversed tracking error from experiment 1, $G(\hat a + \alpha \tilde{e})$. This allows us to compute an estimate of the gradient for a quadratic cost model.

Above, $\tilde{\cdot}$ denotes time reversal and $\hat{\cdot}$ denotes the mean of a signal. $G(a)$ denotes applying dynamic system $G$ to signal $a$, i.e., performing an experiment on $G$ with $a$ as input.

The update to the ILC adjustment signal ``a`` is then computed as
```math
a_{k+1} = a_k - \frac{\beta}{\alpha} \widetilde{\big( G(\hat a + \alpha \tilde{e}) - G(\hat a) \big)}
```
The parameter $\alpha$ controls the size of the "finite-difference perturbation" and $\beta$ is the learning step size. 


Experiment number 3 that uses the time-reversed tracking error gives the algorithm "foresight" in a similar fashion to how [HeuristicILC](@ref) uses a non-causal learning-rate filter ``L``.

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

alg = ModelFreeILC(1, 500)
sol = ilc(prob, alg; actual, iters=10)
plot(sol)
```
The result looks good, and since this algorithm is completely model free, we ran it directly on the "actual" dynamics without considering how it performs when using the model, like we did in the other examples. Keep in mind that this algorithm internally performs 3 rollouts (experiments) for each ILC iteration, so the experimental complexity is 3x of the other algorithms. To show all the intermediate input signals, we use the keyword argument `store_all=true` when calling `ilc`.
    
```@example MODELFREE_ILC
sol = ilc(prob, alg; actual, iters=10, store_all=true)
plot(sol)
```
We now see the total number of rollouts performed. The tracking error now appears to have a non-monotonic behavior, which is expected since the intermediate rollouts apply a signal designed to estimate a gradient rather than minimizing the tracking error.



## Docstring
    
```@docs
ModelFreeILC
```